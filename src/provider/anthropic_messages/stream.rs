use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;

use crate::error::LLMError;
use crate::http::HttpBodyStream;
use crate::provider::ChatStream;
use crate::types::{
    ChatChunk, ChatEvent, ContentDelta, MessageDelta, ProviderMetadata, Role, TokenUsage,
};

use super::response::{convert_finish_reason, convert_usage};

pub(crate) fn create_stream(
    body: HttpBodyStream,
    provider: &'static str,
    endpoint: String,
) -> ChatStream {
    Box::pin(AnthropicSseStream::new(body, provider, endpoint))
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

struct AnthropicSseStream {
    body: HttpBodyStream,
    buffer: Vec<u8>,
    data_lines: Vec<Vec<u8>>,
    pending: VecDeque<Result<ChatChunk, LLMError>>,
    provider: &'static str,
    endpoint: String,
    stream_closed: bool,
    done_received: bool,
}

impl AnthropicSseStream {
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

        // Each Anthropic Messages SSE data chunk corresponds to a JSON event payload.
        let value: Value = serde_json::from_str(&data).map_err(|err| LLMError::Provider {
            provider: self.provider,
            message: format!("failed to parse stream event: {err}"),
        })?;
        let event_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let is_terminal = event_type == "message_stop";
        if is_terminal {
            self.done_received = true;
        }

        let chunk = convert_stream_event(value, self.provider, &self.endpoint, is_terminal)?;
        self.pending.push_back(Ok(chunk));
        Ok(())
    }
}

impl Stream for AnthropicSseStream {
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

fn convert_stream_event(
    event: Value,
    provider: &'static str,
    endpoint: &str,
    is_terminal: bool,
) -> Result<ChatChunk, LLMError> {
    let mut events = Vec::new();
    let mut usage: Option<TokenUsage> = None;

    if let Some(kind) = event.get("type").and_then(|v| v.as_str()) {
        match kind {
            "content_block_delta" => {
                if let Some(delta) = event.get("delta") {
                    if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                        let index =
                            event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        events.push(ChatEvent::MessageDelta(MessageDelta {
                            index,
                            role: Some(Role::assistant()),
                            content: vec![ContentDelta::Text {
                                text: text.to_string(),
                            }],
                            finish_reason: None,
                        }));
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = event.get("delta") {
                    if let Some(usage_obj) = delta.get("usage") {
                        // Attempt to deserialize the usage object into AnthropicUsage and reuse convert_usage.
                        if let Ok(anthropic_usage) = serde_json::from_value::<
                            super::types::AnthropicUsage,
                        >(usage_obj.clone())
                        {
                            usage = Some(convert_usage(&anthropic_usage));
                        }
                    }

                    if let Some(reason) = delta
                        .get("stop_reason")
                        .and_then(|v| v.as_str())
                        .map(convert_finish_reason)
                    {
                        events.push(ChatEvent::MessageDelta(MessageDelta {
                            index: 0,
                            role: Some(Role::assistant()),
                            content: Vec::new(),
                            finish_reason: Some(reason),
                        }));
                    }
                }
            }
            "message_stop" => {
                // Mark the final chunk via `is_terminal`; no extra event is required here.
            }
            _ => {}
        }
    }

    // Always attach a Custom event with the raw structure to aid debugging and extensions.
    events.push(ChatEvent::Custom {
        data: event.clone(),
    });

    Ok(ChatChunk {
        events,
        usage,
        is_terminal,
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: None,
            endpoint: Some(endpoint.to_string()),
            raw: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_text_delta_event_to_message_delta() {
        let event = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "text": "Once upon a time" }
        });
        let chunk =
            convert_stream_event(event, "anthropic_messages", "endpoint", false).expect("convert");

        assert!(!chunk.is_terminal);
        assert_eq!(chunk.provider.provider, "anthropic_messages");
        assert_eq!(chunk.provider.endpoint.as_deref(), Some("endpoint"));
        assert!(chunk.usage.is_none());
        assert!(!chunk.events.is_empty());
    }

    #[test]
    fn convert_message_delta_event_with_usage_and_stop_reason() {
        let event = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5
                }
            }
        });
        let chunk =
            convert_stream_event(event, "anthropic_messages", "endpoint", false).expect("convert");

        assert!(chunk.usage.is_some());
        let usage = chunk.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
    }
}
