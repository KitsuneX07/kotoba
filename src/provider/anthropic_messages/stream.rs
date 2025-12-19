use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::error::LLMError;
use crate::http::HttpBodyStream;
use crate::provider::ChatStream;
use crate::stream::{StreamDecoder, StreamEvent};
use crate::types::{
    ChatChunk, ChatEvent, ContentDelta, MessageDelta, ProviderMetadata, Role, TokenUsage,
};

use super::response::{convert_finish_reason, convert_usage};

pub(crate) fn create_stream(
    body: HttpBodyStream,
    provider: &'static str,
    endpoint: String,
) -> ChatStream {
    let stream = StreamDecoder::new(body, provider).map(move |event| match event {
        Ok(StreamEvent::Data(data)) => {
            let value: Value = serde_json::from_str(&data).map_err(|err| LLMError::Provider {
                provider,
                message: format!("failed to parse stream event: {err}"),
            })?;
            let event_type = value
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let is_terminal = event_type == "message_stop";
            convert_stream_event(value, provider, &endpoint, is_terminal)
        }
        Ok(StreamEvent::Done) => Ok(ChatChunk {
            events: Vec::new(),
            usage: None,
            is_terminal: true,
            provider: ProviderMetadata {
                provider: provider.to_string(),
                request_id: None,
                endpoint: Some(endpoint.clone()),
                raw: Some(json!({"event": "[DONE]"})),
            },
        }),
        Err(err) => Err(err),
    });
    Box::pin(stream)
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
