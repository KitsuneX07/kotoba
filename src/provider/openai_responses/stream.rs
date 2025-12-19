use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::StreamExt;
use serde_json::json;

use crate::error::LLMError;
use crate::http::HttpBodyStream;
use crate::provider::ChatStream;
use crate::stream::{StreamDecoder, StreamEvent};
use crate::types::{ChatChunk, ChatEvent, ContentDelta, MessageDelta, ProviderMetadata, Role};

use super::response::convert_usage;
use super::types::OpenAiResponsesStreamEvent;

pub(crate) fn create_stream(
    body: HttpBodyStream,
    provider: &'static str,
    endpoint: String,
) -> ChatStream {
    let terminal_emitted = Arc::new(AtomicBool::new(false));
    let stream = StreamDecoder::new(body, provider).filter_map(move |event| {
        let endpoint = endpoint.clone();
        let terminal_flag = Arc::clone(&terminal_emitted);
        async move {
            match event {
                Ok(StreamEvent::Data(data)) => {
                    let parsed: OpenAiResponsesStreamEvent = match serde_json::from_str(&data)
                        .map_err(|err| LLMError::Provider {
                            provider,
                            message: format!("failed to parse Responses stream event: {err}"),
                        }) {
                        Ok(event) => event,
                        Err(err) => return Some(Err(err)),
                    };
                    match convert_stream_event(parsed, provider, &endpoint) {
                        Ok(Some(chunk)) => {
                            if chunk.is_terminal {
                                terminal_flag.store(true, Ordering::Relaxed);
                            }
                            Some(Ok(chunk))
                        }
                        Ok(None) => None,
                        Err(err) => Some(Err(err)),
                    }
                }
                Ok(StreamEvent::Done) => {
                    if terminal_flag.swap(true, Ordering::Relaxed) {
                        None
                    } else {
                        Some(Ok(ChatChunk {
                            events: Vec::new(),
                            usage: None,
                            is_terminal: true,
                            provider: ProviderMetadata {
                                provider: provider.to_string(),
                                request_id: None,
                                endpoint: Some(endpoint.clone()),
                                raw: Some(json!({"event": "[DONE]"})),
                            },
                        }))
                    }
                }
                Err(err) => Some(Err(err)),
            }
        }
    });
    Box::pin(stream)
}

/// Collects the full response body when a streaming request returns a non-2xx status.
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
                // Streaming text events originate from the assistant role.
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
        // Ignore other events (response.created, response.in_progress, output_item.*, etc.).
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai_responses::types::{OpenAiResponsesResponse, OpenAiResponsesUsage};
    use crate::types::ChatEvent;

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

        let chunk = convert_stream_event(event, "openai_responses", "endpoint")
            .expect("convert should succeed");
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

        let chunk = convert_stream_event(event, "openai_responses", "endpoint")
            .expect("convert should succeed");
        let chunk = chunk.expect("chunk should be Some");
        assert!(chunk.is_terminal);
        assert!(chunk.events.is_empty());
        let usage = chunk.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
    }
}
