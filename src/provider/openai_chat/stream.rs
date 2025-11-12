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
use super::types::{OpenAiDeltaContent, OpenAiMessagePart, OpenAiStreamChunk, OpenAiToolCallDelta};

pub(crate) fn create_stream(
    body: HttpBodyStream,
    provider: &'static str,
    endpoint: String,
) -> ChatStream {
    Box::pin(OpenAiSseStream::new(body, provider, endpoint))
}

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

struct OpenAiSseStream {
    body: HttpBodyStream,
    buffer: Vec<u8>,
    data_lines: Vec<Vec<u8>>,
    pending: VecDeque<Result<ChatChunk, LLMError>>,
    provider: &'static str,
    endpoint: String,
    stream_closed: bool,
    done_received: bool,
}

impl OpenAiSseStream {
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
            let chunk: OpenAiStreamChunk =
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

impl Stream for OpenAiSseStream {
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
    chunk: OpenAiStreamChunk,
    provider: &'static str,
    endpoint: &str,
) -> Result<ChatChunk, LLMError> {
    let mut events = Vec::new();
    for choice in &chunk.choices {
        if let Some(delta) = &choice.delta {
            if delta.role.is_some() || delta.content.is_some() || choice.finish_reason.is_some() {
                let content_updates = match &delta.content {
                    Some(OpenAiDeltaContent::Parts(parts)) => convert_content_delta(parts)?,
                    Some(OpenAiDeltaContent::Text(text)) => {
                        if text.is_empty() {
                            Vec::new()
                        } else {
                            vec![ContentDelta::Text { text: text.clone() }]
                        }
                    }
                    None => Vec::new(),
                };
                let message_delta = MessageDelta {
                    index: choice.index,
                    role: delta.role.clone().map(crate::types::Role),
                    content: content_updates,
                    finish_reason: choice.finish_reason.as_deref().map(convert_finish_reason),
                };
                if message_delta.role.is_some()
                    || !message_delta.content.is_empty()
                    || message_delta.finish_reason.is_some()
                {
                    events.push(ChatEvent::MessageDelta(message_delta));
                }
            }
            if let Some(tool_calls) = &delta.tool_calls {
                for tool_call in tool_calls {
                    let delta = convert_tool_call_delta_event(
                        tool_call,
                        choice.index,
                        choice.finish_reason.as_deref(),
                    )?;
                    events.push(ChatEvent::ToolCallDelta(delta));
                }
            }
        }
    }
    let usage = chunk.usage.clone().map(convert_usage);
    let raw = serde_json::to_value(&chunk).ok();
    Ok(ChatChunk {
        events,
        usage,
        is_terminal: false,
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: None,
            endpoint: Some(endpoint.to_string()),
            raw,
        },
    })
}

fn convert_content_delta(parts: &[OpenAiMessagePart]) -> Result<Vec<ContentDelta>, LLMError> {
    let mut deltas = Vec::new();
    for part in parts {
        match part.kind.as_str() {
            "text" | "input_text" => {
                if let Some(text) = &part.text {
                    if !text.is_empty() {
                        deltas.push(ContentDelta::Text { text: text.clone() });
                    }
                }
            }
            _ => {
                let value =
                    serde_json::to_value(part).unwrap_or_else(|_| json!({ "type": part.kind }));
                deltas.push(ContentDelta::Json { value });
            }
        }
    }
    Ok(deltas)
}

fn convert_tool_call_delta_event(
    delta: &OpenAiToolCallDelta,
    fallback_index: usize,
    finish_reason: Option<&str>,
) -> Result<ToolCallDelta, LLMError> {
    let index = delta.index.unwrap_or(fallback_index);
    let (name, arguments) = delta
        .function
        .as_ref()
        .map(|f| (f.name.clone(), f.arguments.clone()))
        .unwrap_or((None, None));
    let kind = match delta.kind.as_deref() {
        Some("function") => Some(ToolCallKind::Function),
        _ => None,
    };
    Ok(ToolCallDelta {
        index,
        id: delta.id.clone(),
        name,
        arguments_delta: arguments,
        kind,
        is_finished: matches!(finish_reason, Some("tool_calls")),
    })
}
