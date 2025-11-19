use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::json;

use crate::error::LLMError;
use crate::http::HttpBodyStream;
use crate::provider::ChatStream;
use crate::types::{
    ChatChunk, ChatEvent, ContentDelta, MessageDelta, ProviderMetadata, Role,
};

use super::response::convert_usage;
use super::types::OpenAiResponsesStreamEvent;

pub(crate) fn create_stream(
    body: HttpBodyStream,
    provider: &'static str,
    endpoint: String,
) -> ChatStream {
    Box::pin(OpenAiResponsesSseStream::new(body, provider, endpoint))
}

/// 当流式请求返回非 2xx 状态码时收集 body 用于错误信息
pub(crate) async fn collect_stream_text(
    mut body: HttpBodyStream,
    provider: &'static str,
) -> Result<String, LLMError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = body.next().await {
        bytes.extend_from_slice(&chunk?);
    }
    String::from_utf8(bytes).map_err(|err| LLMError::Provider {
        provider,
        message: format!("failed to decode stream error body: {err}"),
    })
}

struct OpenAiResponsesSseStream {
    body: HttpBodyStream,
    buffer: Vec<u8>,
    data_lines: Vec<Vec<u8>>,
    pending: VecDeque<Result<ChatChunk, LLMError>>,
    provider: &'static str,
    endpoint: String,
    stream_closed: bool,
    done_received: bool,
}

impl OpenAiResponsesSseStream {
    fn new(body: HttpBodyStream, provider: &'static str, endpoint: String) -> Self {
        Self {
            body,
            buffer: Vec::new(),
            data_lines: Vec::new(),
            pending: VecDeque::new(),
            provider,
            endpoint,
            stream_closed: false,
            done_received: false,
        }
    }

    fn handle_line(&mut self, line: Vec<u8>) {
        if line.starts_with(b"data:") {
            let mut data = line[5..].to_vec();
            if let Some(first) = data.first() {
                if *first == b' ' {
                    data.remove(0);
                }
            }
            self.data_lines.push(data);
        }
    }

    fn flush_event(&mut self) -> Result<(), LLMError> {
        if self.data_lines.is_empty() {
            return Ok(());
        }
        let mut joined = Vec::new();
        for (idx, mut segment) in self.data_lines.drain(..).enumerate() {
            if idx > 0 {
                joined.push(b'\n');
            }
            joined.append(&mut segment);
        }
        if joined.is_empty() {
            return Ok(());
        }
        let data = String::from_utf8(joined).map_err(|err| LLMError::Provider {
            provider: self.provider,
            message: format!("invalid UTF-8 in stream chunk: {err}"),
        })?;
        if data.trim() == "[DONE]" {
            // 如果已经收到 completed 事件，则忽略额外的 [DONE]
            if !self.done_received {
                let chunk = ChatChunk {
                    events: Vec::new(),
                    usage: None,
                    is_terminal: true,
                    provider: ProviderMetadata {
                        provider: self.provider.to_string(),
                        request_id: None,
                        endpoint: Some(self.endpoint.clone()),
                        raw: Some(json!({"event": "[DONE]"})),
                    },
                };
                self.pending.push_back(Ok(chunk));
                self.done_received = true;
            }
        } else {
            let event: OpenAiResponsesStreamEvent =
                serde_json::from_str(&data).map_err(|err| LLMError::Provider {
                    provider: self.provider,
                    message: format!("failed to parse Responses stream event: {err}"),
                })?;
            if let Some(chunk) = convert_stream_event(event, self.provider, &self.endpoint)? {
                if chunk.is_terminal {
                    self.done_received = true;
                }
                self.pending.push_back(Ok(chunk));
            }
        }
        Ok(())
    }
}

impl Stream for OpenAiResponsesSseStream {
    type Item = Result<ChatChunk, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(item) = this.pending.pop_front() {
            return Poll::Ready(Some(item));
        }
        loop {
            if this.stream_closed {
                if !this.buffer.is_empty() {
                    let line = this.buffer.drain(..).collect::<Vec<_>>();
                    this.handle_line(line);
                    if let Err(err) = this.flush_event() {
                        return Poll::Ready(Some(Err(err)));
                    }
                    if let Some(item) = this.pending.pop_front() {
                        return Poll::Ready(Some(item));
                    }
                }
                return Poll::Ready(None);
            }
            match this.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(chunk_result)) => match chunk_result {
                    Ok(bytes) => {
                        this.buffer.extend_from_slice(&bytes);
                        while let Some(pos) = find_newline(&this.buffer) {
                            let mut line: Vec<u8> = this.buffer.drain(..=pos).collect();
                            if line.last() == Some(&b'\n') {
                                line.pop();
                            }
                            if line.last() == Some(&b'\r') {
                                line.pop();
                            }
                            if line.is_empty() {
                                if let Err(err) = this.flush_event() {
                                    return Poll::Ready(Some(Err(err)));
                                }
                                if let Some(item) = this.pending.pop_front() {
                                    return Poll::Ready(Some(item));
                                }
                            } else {
                                this.handle_line(line);
                            }
                        }
                        if let Some(item) = this.pending.pop_front() {
                            return Poll::Ready(Some(item));
                        }
                    }
                    Err(err) => return Poll::Ready(Some(Err(err))),
                },
                Poll::Ready(None) => {
                    this.stream_closed = true;
                    if let Err(err) = this.flush_event() {
                        return Poll::Ready(Some(Err(err)));
                    }
                    return this
                        .pending
                        .pop_front()
                        .map_or(Poll::Ready(None), |item| Poll::Ready(Some(item)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn find_newline(buffer: &[u8]) -> Option<usize> {
    buffer.iter().position(|b| *b == b'\n')
}

fn convert_stream_event(
    event: OpenAiResponsesStreamEvent,
    provider: &'static str,
    endpoint: &str,
) -> Result<Option<ChatChunk>, LLMError> {
    match event.event_type.as_str() {
        "response.output_text.delta" => {
            let index = event.output_index.unwrap_or(0);
            let delta = event.delta.as_deref().unwrap_or("").to_string();
            if delta.is_empty() {
                return Ok(None);
            }
            let message_delta = MessageDelta {
                index,
                // Responses 流式文本默认来自 assistant 角色
                role: Some(Role::assistant()),
                content: vec![ContentDelta::Text { text: delta }],
                finish_reason: None,
            };
            let raw = serde_json::to_value(event).ok();
            Ok(Some(ChatChunk {
                events: vec![ChatEvent::MessageDelta(message_delta)],
                usage: None,
                is_terminal: false,
                provider: ProviderMetadata {
                    provider: provider.to_string(),
                    request_id: None,
                    endpoint: Some(endpoint.to_string()),
                    raw,
                },
            }))
        }
        "response.completed" => {
            let response = event.response.ok_or_else(|| LLMError::Provider {
                provider,
                message: "response.completed event missing response".to_string(),
            })?;
            let raw = serde_json::to_value(&response).ok();
            let usage = response.usage.map(convert_usage);
            Ok(Some(ChatChunk {
                events: Vec::new(),
                usage,
                is_terminal: true,
                provider: ProviderMetadata {
                    provider: provider.to_string(),
                    request_id: None,
                    endpoint: Some(endpoint.to_string()),
                    raw,
                },
            }))
        }
        // 其他事件先忽略（response.created / response.in_progress / output_item.* 等），必要时扩展为 Custom 事件
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatEvent;
    use crate::provider::openai_responses::types::{OpenAiResponsesResponse, OpenAiResponsesUsage};

    #[test]
    fn convert_output_text_delta_event() {
        let event = OpenAiResponsesStreamEvent {
            event_type: "response.output_text.delta".to_string(),
            response: None,
            output_index: Some(0),
            content_index: Some(0),
            delta: Some("hello".to_string()),
            item_id: Some("msg_1".to_string()),
            item: None,
            part: None,
        };

        let chunk =
            convert_stream_event(event, "openai_responses", "endpoint").expect("convert should succeed");
        let chunk = chunk.expect("chunk should be Some");
        assert!(!chunk.is_terminal);
        assert_eq!(chunk.provider.provider, "openai_responses");
        assert_eq!(chunk.provider.endpoint.as_deref(), Some("endpoint"));
        assert!(chunk.usage.is_none());
        assert_eq!(chunk.events.len(), 1);
        match &chunk.events[0] {
            ChatEvent::MessageDelta(delta) => {
                assert_eq!(delta.index, 0);
                assert_eq!(delta.role.as_ref().map(|r| r.0.as_str()), Some("assistant"));
                assert_eq!(delta.content.len(), 1);
                match &delta.content[0] {
                    ContentDelta::Text { text } => assert_eq!(text, "hello"),
                    other => panic!("unexpected content delta: {other:?}"),
                }
            }
            other => panic!("unexpected chat event: {other:?}"),
        }
    }

    #[test]
    fn convert_completed_event_to_terminal_chunk() {
        let event = OpenAiResponsesStreamEvent {
            event_type: "response.completed".to_string(),
            response: Some(OpenAiResponsesResponse {
                id: "resp_1".to_string(),
                object: "response".to_string(),
                created_at: Some(1),
                status: Some("completed".to_string()),
                error: None,
                incomplete_details: None,
                instructions: None,
                max_output_tokens: None,
                model: "gpt-4.1".to_string(),
                output: Vec::new(),
                parallel_tool_calls: Some(true),
                previous_response_id: None,
                reasoning: None,
                store: Some(true),
                temperature: Some(1.0),
                text: None,
                tool_choice: None,
                tools: None,
                top_p: Some(1.0),
                truncation: Some("disabled".to_string()),
                usage: Some(OpenAiResponsesUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    total_tokens: Some(15),
                    input_tokens_details: None,
                    output_tokens_details: None,
                }),
                user: None,
                metadata: None,
            }),
            output_index: None,
            content_index: None,
            delta: None,
            item_id: None,
            item: None,
            part: None,
        };

        let chunk =
            convert_stream_event(event, "openai_responses", "endpoint").expect("convert should succeed");
        let chunk = chunk.expect("chunk should be Some");
        assert!(chunk.is_terminal);
        assert!(chunk.events.is_empty());
        let usage = chunk.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
    }
}
