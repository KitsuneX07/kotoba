use futures_util::StreamExt;
use serde_json::json;

use crate::error::LLMError;
use crate::http::HttpBodyStream;
use crate::provider::ChatStream;
use crate::stream::{StreamDecoder, StreamEvent};
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
    let stream = StreamDecoder::new(body, provider).map(move |event| match event {
        Ok(StreamEvent::Data(data)) => {
            let chunk: OpenAiStreamChunk =
                serde_json::from_str(&data).map_err(|err| LLMError::Provider {
                    provider,
                    message: format!("failed to parse stream chunk: {err}"),
                })?;
            convert_stream_chunk(chunk, provider, &endpoint)
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

#[cfg(test)]
mod tests {
    use super::super::types::{
        OpenAiDeltaContent, OpenAiMessagePart, OpenAiStreamChoice, OpenAiStreamChunk,
        OpenAiStreamDelta, OpenAiToolCallDelta, OpenAiToolFunctionDelta, OpenAiUsage,
    };
    use super::*;
    use crate::types::{ChatEvent, ContentDelta, FinishReason};

    /// Stream chunk that only contains a text delta.
    #[test]
    fn convert_stream_chunk_with_text_delta() {
        let chunk = OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                index: 0,
                delta: Some(OpenAiStreamDelta {
                    role: Some("assistant".to_string()),
                    content: Some(OpenAiDeltaContent::Text("hello".to_string())),
                    tool_calls: None,
                }),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: Some(1),
                completion_tokens: Some(2),
                total_tokens: Some(3),
                reasoning_tokens: Some(0),
            }),
        };

        let chat_chunk =
            convert_stream_chunk(chunk, "openai_chat", "endpoint").expect("convert should succeed");

        assert!(!chat_chunk.is_terminal);
        assert_eq!(chat_chunk.provider.provider, "openai_chat");
        assert_eq!(chat_chunk.provider.endpoint.as_deref(), Some("endpoint"));

        let usage = chat_chunk.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(1));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(3));

        // Emit a single MessageDelta event.
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

    /// Converts content delta events that arrive as message parts.
    #[test]
    fn convert_content_delta_from_parts() {
        let parts = vec![
            OpenAiMessagePart {
                kind: "text".to_string(),
                text: Some("hi".to_string()),
                image_url: None,
                extra: Default::default(),
            },
            OpenAiMessagePart {
                kind: "custom".to_string(),
                text: None,
                image_url: None,
                extra: [("x".to_string(), serde_json::json!(1))]
                    .into_iter()
                    .collect(),
            },
        ];

        let deltas = convert_content_delta(&parts).expect("convert should succeed");
        assert_eq!(deltas.len(), 2);
        match &deltas[0] {
            ContentDelta::Text { text } => assert_eq!(text, "hi"),
            other => panic!("unexpected first delta: {other:?}"),
        }
        match &deltas[1] {
            ContentDelta::Json { value } => {
                assert_eq!(value["type"], serde_json::json!("custom"));
                assert_eq!(value["x"], serde_json::json!(1));
            }
            other => panic!("unexpected second delta: {other:?}"),
        }
    }

    /// Converts tool-call delta fragments into [`ToolCallDelta`] events.
    #[test]
    fn convert_tool_call_delta_event_basic() {
        let delta = OpenAiToolCallDelta {
            index: Some(3),
            id: Some("call_1".to_string()),
            kind: Some("function".to_string()),
            function: Some(OpenAiToolFunctionDelta {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Boston"}"#.to_string()),
            }),
        };

        let mapped = convert_tool_call_delta_event(&delta, 0, Some("tool_calls"))
            .expect("convert should succeed");

        assert_eq!(mapped.index, 3);
        assert_eq!(mapped.id.as_deref(), Some("call_1"));
        assert_eq!(mapped.name.as_deref(), Some("get_weather"));
        assert_eq!(
            mapped.arguments_delta.as_deref(),
            Some(r#"{"city":"Boston"}"#)
        );
        assert!(matches!(mapped.kind, Some(ToolCallKind::Function)));
        assert!(mapped.is_finished);
    }
}
