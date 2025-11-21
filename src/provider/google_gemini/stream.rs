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
    ChatChunk, ChatEvent, ContentDelta, MessageDelta, ProviderMetadata, ToolCallDelta, ToolCallKind,
};

use super::response::{convert_finish_reason, convert_usage};
use super::types::GeminiGenerateContentResponse;

/// Wraps the raw HTTP stream into a [`ChatStream`].
pub(crate) fn create_stream(
    body: HttpBodyStream,
    provider: &'static str,
    endpoint: String,
) -> ChatStream {
    Box::pin(GeminiSseStream::new(body, provider, endpoint))
}

/// Collects the entire stream body when errors occur to build rich error messages.
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

struct GeminiSseStream {
    body: HttpBodyStream,
    buffer: Vec<u8>,
    data_lines: Vec<Vec<u8>>,
    pending: VecDeque<Result<ChatChunk, LLMError>>,
    provider: &'static str,
    endpoint: String,
    stream_closed: bool,
    done_received: bool,
}

impl GeminiSseStream {
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
            self.done_received = true;
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
        } else {
            let chunk: GeminiGenerateContentResponse =
                serde_json::from_str(&data).map_err(|err| LLMError::Provider {
                    provider: self.provider,
                    message: format!("failed to parse stream chunk: {err}"),
                })?;
            let chat_chunk = convert_stream_chunk(chunk, self.provider, &self.endpoint)?;
            self.pending.push_back(Ok(chat_chunk));
        }
        Ok(())
    }
}

impl Stream for GeminiSseStream {
    type Item = Result<ChatChunk, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(item) = this.pending.pop_front() {
            return Poll::Ready(Some(item));
        }
        if this.done_received && this.pending.is_empty() {
            return Poll::Ready(None);
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

fn convert_stream_chunk(
    chunk: GeminiGenerateContentResponse,
    provider: &'static str,
    endpoint: &str,
) -> Result<ChatChunk, LLMError> {
    use crate::types::Role;

    let mut events = Vec::new();

    for (default_index, candidate) in chunk.candidates.iter().enumerate() {
        let index = candidate.index.unwrap_or(default_index);
        if let Some(content) = &candidate.content {
            // Map Gemini roles to unified roles (`model` -> assistant).
            let role = content
                .role
                .as_deref()
                .map(|r| match r {
                    "model" => Role::assistant(),
                    other => Role(other.to_string()),
                })
                .unwrap_or_else(Role::assistant);

            let mut content_deltas = Vec::new();

            for part in &content.parts {
                // Function-call delta translates into a [`ToolCallDelta`].
                if let Some(call) = &part.function_call {
                    let args_str = match serde_json::to_string(&call.args) {
                        Ok(s) => s,
                        Err(_) => call.args.to_string(),
                    };
                    let delta = ToolCallDelta {
                        index,
                        id: None,
                        name: Some(call.name.clone()),
                        arguments_delta: Some(args_str),
                        kind: Some(ToolCallKind::Function),
                        // Gemini lacks a dedicated tool_calls finish reason, so treat it as finished.
                        is_finished: true,
                    };
                    events.push(ChatEvent::ToolCallDelta(delta));
                    continue;
                }

                // Text fragments become [`ContentDelta::Text`].
                if let Some(text) = &part.text {
                    if !text.is_empty() {
                        content_deltas.push(ContentDelta::Text { text: text.clone() });
                        continue;
                    }
                }

                // Other multimodal/meta parts become JSON deltas for consumers to parse.
                let value = serde_json::to_value(part).unwrap_or_else(|_| json!({}));
                content_deltas.push(ContentDelta::Json { value });
            }

            if !content_deltas.is_empty() || candidate.finish_reason.is_some() {
                let message_delta = MessageDelta {
                    index,
                    role: Some(role),
                    content: content_deltas,
                    finish_reason: candidate
                        .finish_reason
                        .as_deref()
                        .map(convert_finish_reason),
                };
                events.push(ChatEvent::MessageDelta(message_delta));
            }
        }
    }

    let usage = chunk.usage_metadata.as_ref().map(convert_usage);
    let raw = serde_json::to_value(&chunk).ok();
    Ok(ChatChunk {
        events,
        usage,
        is_terminal: false,
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: chunk.response_id,
            endpoint: Some(endpoint.to_string()),
            raw,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::types::{GeminiCandidate, GeminiContent, GeminiPart, GeminiUsageMetadata};
    use super::*;
    use crate::types::{ChatEvent, ContentDelta, FinishReason};

    /// Ensures pure text deltas convert into a single [`MessageDelta`].
    #[test]
    fn convert_stream_chunk_with_text_delta() {
        let chunk = GeminiGenerateContentResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    parts: vec![GeminiPart {
                        text: Some("hello".to_string()),
                        inline_data: None,
                        file_data: None,
                        function_call: None,
                        function_response: None,
                        executable_code: None,
                        code_execution_result: None,
                        extra: Default::default(),
                    }],
                    role: Some("model".to_string()),
                    extra: Default::default(),
                }),
                finish_reason: Some("STOP".to_string()),
                index: Some(0),
                extra: Default::default(),
            }],
            prompt_feedback: None,
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(1),
                cached_content_token_count: None,
                candidates_token_count: Some(2),
                total_token_count: Some(3),
                tool_use_prompt_token_count: None,
                thoughts_token_count: None,
                extra: Default::default(),
            }),
            model_version: Some("gemini-2.0-flash".to_string()),
            response_id: None,
            extra: Default::default(),
        };

        let chat_chunk =
            convert_stream_chunk(chunk, "google_gemini", "endpoint").expect("convert succeeds");

        assert!(!chat_chunk.is_terminal);
        assert_eq!(chat_chunk.provider.provider, "google_gemini");
        assert_eq!(chat_chunk.provider.endpoint.as_deref(), Some("endpoint"));

        let usage = chat_chunk.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(1));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(3));

        assert_eq!(chat_chunk.events.len(), 1);
        match &chat_chunk.events[0] {
            ChatEvent::MessageDelta(delta) => {
                assert_eq!(delta.index, 0);
                assert!(matches!(
                    delta.role.as_ref().map(|r| r.0.as_str()),
                    Some("assistant")
                ));
                assert!(matches!(delta.finish_reason, Some(FinishReason::Stop)));
                assert_eq!(delta.content.len(), 1);
                match &delta.content[0] {
                    ContentDelta::Text { text } => assert_eq!(text, "hello"),
                    other => panic!("unexpected content delta: {other:?}"),
                }
            }
            other => panic!("unexpected chat event: {other:?}"),
        }
    }
}
