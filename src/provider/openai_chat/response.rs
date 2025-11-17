use serde_json::{Value, json};

use crate::error::LLMError;
use crate::types::{
    ChatResponse, ContentPart, FinishReason, Message, OutputItem, ProviderMetadata, Role,
    TextContent, TokenUsage, ToolCall, ToolCallKind,
};

use super::types::{
    OpenAiChatResponse, OpenAiMessageContent, OpenAiMessagePart, OpenAiResponseMessage,
    OpenAiToolCallResponse, OpenAiUsage,
};

pub(crate) fn map_response(
    resp: OpenAiChatResponse,
    provider: &'static str,
    endpoint: String,
) -> Result<ChatResponse, LLMError> {
    let raw = serde_json::to_value(&resp).ok();
    let mut outputs = Vec::new();
    for choice in &resp.choices {
        if let Some(message) = &choice.message {
            let (msg, tool_calls) = convert_response_message(message.clone())?;
            outputs.push(OutputItem::Message {
                message: msg,
                index: choice.index,
            });
            for call in tool_calls {
                outputs.push(OutputItem::ToolCall {
                    call,
                    index: choice.index,
                });
            }
        }
    }
    let finish_reason = resp
        .choices
        .iter()
        .find_map(|choice| choice.finish_reason.as_deref().map(convert_finish_reason));
    let usage = resp.usage.clone().map(convert_usage);
    let model = resp.model.clone();
    Ok(ChatResponse {
        outputs,
        usage,
        finish_reason,
        model: Some(model),
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: None,
            endpoint: Some(endpoint),
            raw,
        },
    })
}

fn convert_response_message(
    message: OpenAiResponseMessage,
) -> Result<(Message, Vec<ToolCall>), LLMError> {
    let role = message
        .role
        .clone()
        .map(Role)
        .unwrap_or_else(|| Role("assistant".to_string()));
    let content = match &message.content {
        None => Vec::new(),
        Some(OpenAiMessageContent::Text(text)) => {
            vec![ContentPart::Text(TextContent { text: text.clone() })]
        }
        Some(OpenAiMessageContent::Parts(parts)) => parts
            .iter()
            .cloned()
            .map(convert_content_part_response)
            .collect::<Result<Vec<_>, _>>()?,
    };
    let tool_calls = message
        .tool_calls
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(convert_tool_call_response)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        Message {
            role,
            name: message.name.clone(),
            content,
            metadata: None,
        },
        tool_calls,
    ))
}

pub(crate) fn convert_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        "function_call" => FinishReason::FunctionCall,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(crate) fn convert_usage(usage: OpenAiUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        total_tokens: usage.total_tokens,
        details: None,
    }
}

fn convert_tool_call_response(call: OpenAiToolCallResponse) -> Result<ToolCall, LLMError> {
    if call.kind != "function" {
        return Err(LLMError::Provider {
            provider: "openai_chat",
            message: format!("unsupported tool type {}", call.kind),
        });
    }
    let (name, arguments) = if let Some(function) = call.function {
        (function.name.unwrap_or_default(), function.arguments)
    } else {
        (String::new(), None)
    };
    let args_value = if let Some(args) = arguments {
        serde_json::from_str(&args).unwrap_or(Value::String(args))
    } else {
        Value::Null
    };
    Ok(ToolCall {
        id: call.id,
        name,
        arguments: args_value,
        kind: ToolCallKind::Function,
    })
}

fn convert_content_part_response(part: OpenAiMessagePart) -> Result<ContentPart, LLMError> {
    match part.kind.as_str() {
        "text" => Ok(ContentPart::Text(TextContent {
            text: part.text.unwrap_or_default(),
        })),
        "image_url" => {
            if let Some(url) = part.image_url {
                Ok(ContentPart::Image(crate::types::ImageContent {
                    source: crate::types::ImageSource::Url { url: url.url },
                    detail: url.detail.and_then(|d| match d.as_str() {
                        "low" => Some(crate::types::ImageDetail::Low),
                        "high" => Some(crate::types::ImageDetail::High),
                        "auto" => Some(crate::types::ImageDetail::Auto),
                        _ => None,
                    }),
                    metadata: None,
                }))
            } else {
                Err(LLMError::Provider {
                    provider: "openai_chat",
                    message: "image_url part missing url".to_string(),
                })
            }
        }
        _ => {
            let value = serde_json::to_value(part).unwrap_or_else(|_| json!({}));
            Ok(ContentPart::Data { data: value })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentPart, ImageDetail, ImageSource};

    fn sample_response_text() -> OpenAiChatResponse {
        OpenAiChatResponse {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion".to_string(),
            created: Some(1),
            model: "gpt-4.1".to_string(),
            choices: vec![super::super::types::OpenAiResponseChoice {
                index: 0,
                message: Some(OpenAiResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some(OpenAiMessageContent::Text(
                        "hello world".to_string(),
                    )),
                    name: None,
                    tool_calls: None,
                }),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                total_tokens: Some(15),
                reasoning_tokens: Some(0),
            }),
            service_tier: Some("default".to_string()),
            system_fingerprint: None,
        }
    }

    #[test]
    fn map_response_text_only() {
        let resp = sample_response_text();
        let mapped =
            map_response(resp, "openai_chat", "https://api.openai.com/v1/chat/completions".into())
                .expect("map_response should succeed");

        assert_eq!(mapped.model.as_deref(), Some("gpt-4.1"));
        assert!(matches!(mapped.finish_reason, Some(FinishReason::Stop)));
        assert_eq!(mapped.provider.provider, "openai_chat");
        assert_eq!(
            mapped.provider.endpoint.as_deref(),
            Some("https://api.openai.com/v1/chat/completions")
        );

        // 有一个 Message 输出
        assert_eq!(mapped.outputs.len(), 1);
        match &mapped.outputs[0] {
            OutputItem::Message { message, index } => {
                assert_eq!(*index, 0);
                assert_eq!(message.role.0, "assistant");
                assert_eq!(message.name, None);
                // 内容是单一文本块
                assert_eq!(message.content.len(), 1);
                match &message.content[0] {
                    ContentPart::Text(TextContent { text }) => {
                        assert_eq!(text, "hello world");
                    }
                    other => panic!("unexpected content part: {other:?}"),
                }
            }
            other => panic!("unexpected output item: {other:?}"),
        }

        // usage 映射
        let usage = mapped.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
        assert_eq!(usage.reasoning_tokens, Some(0));
    }

    #[test]
    fn map_response_with_tool_calls() {
        use super::super::types::{OpenAiResponseChoice, OpenAiToolCallResponse};

        let resp = OpenAiChatResponse {
            id: "chatcmpl-2".to_string(),
            object: "chat.completion".to_string(),
            created: None,
            model: "gpt-4.1".to_string(),
            choices: vec![OpenAiResponseChoice {
                index: 0,
                message: Some(OpenAiResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some(OpenAiMessageContent::Parts(vec![
                        OpenAiMessagePart {
                            kind: "text".to_string(),
                            text: Some("".to_string()),
                            image_url: None,
                            extra: Default::default(),
                        },
                    ])),
                    name: None,
                    tool_calls: Some(vec![OpenAiToolCallResponse {
                        id: Some("call_1".to_string()),
                        kind: "function".to_string(),
                        function: Some(super::super::types::OpenAiToolFunction {
                            name: Some("get_weather".to_string()),
                            arguments: Some(r#"{"location":"Boston, MA"}"#.to_string()),
                        }),
                    }]),
                }),
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
            service_tier: None,
            system_fingerprint: None,
        };

        let mapped =
            map_response(resp, "openai_chat", "endpoint".into()).expect("map_response should work");

        // 既有 Message 又有 ToolCall 两个输出
        assert_eq!(mapped.outputs.len(), 2);
        let mut saw_message = false;
        let mut saw_tool_call = false;
        for item in &mapped.outputs {
            match item {
                OutputItem::Message { message, .. } => {
                    saw_message = true;
                    assert_eq!(message.role.0, "assistant");
                }
                OutputItem::ToolCall { call, .. } => {
                    saw_tool_call = true;
                    assert_eq!(call.id.as_deref(), Some("call_1"));
                    assert_eq!(call.name, "get_weather");
                    // 参数解析为 JSON 对象
                    assert_eq!(
                        call.arguments["location"],
                        json!("Boston, MA")
                    );
                    assert_eq!(call.kind, ToolCallKind::Function);
                }
                _ => {}
            }
        }
        assert!(saw_message && saw_tool_call);
        assert!(matches!(
            mapped.finish_reason,
            Some(FinishReason::ToolCalls)
        ));
    }

    #[test]
    fn convert_content_part_response_text_and_image() {
        // 文本
        let part_text = OpenAiMessagePart {
            kind: "text".to_string(),
            text: Some("hi".to_string()),
            image_url: None,
            extra: Default::default(),
        };
        let mapped = convert_content_part_response(part_text).expect("text part should map");
        match mapped {
            ContentPart::Text(TextContent { text }) => assert_eq!(text, "hi"),
            other => panic!("unexpected content: {other:?}"),
        }

        // 图像
        let part_image = OpenAiMessagePart {
            kind: "image_url".to_string(),
            text: None,
            image_url: Some(super::super::types::OpenAiImageUrl {
                url: "https://example.com/img.png".to_string(),
                detail: Some("high".to_string()),
            }),
            extra: Default::default(),
        };
        let mapped = convert_content_part_response(part_image).expect("image part should map");
        match mapped {
            ContentPart::Image(img) => {
                match img.source {
                    ImageSource::Url { url } => {
                        assert_eq!(url, "https://example.com/img.png");
                    }
                    _ => panic!("unexpected image source"),
                }
                assert!(matches!(img.detail, Some(ImageDetail::High)));
            }
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn convert_content_part_response_unknown_type_becomes_data() {
        let part = OpenAiMessagePart {
            kind: "custom_type".to_string(),
            text: None,
            image_url: None,
            extra: [("field".to_string(), json!(1))].into_iter().collect(),
        };
        let mapped = convert_content_part_response(part).expect("custom part should map");
        match mapped {
            ContentPart::Data { data } => {
                assert_eq!(data["type"], json!("custom_type"));
                assert_eq!(data["field"], json!(1));
            }
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn convert_finish_reason_and_usage() {
        assert!(matches!(
            convert_finish_reason("stop"),
            FinishReason::Stop
        ));
        assert!(matches!(
            convert_finish_reason("length"),
            FinishReason::Length
        ));
        assert!(matches!(
            convert_finish_reason("tool_calls"),
            FinishReason::ToolCalls
        ));
        assert!(matches!(
            convert_finish_reason("content_filter"),
            FinishReason::ContentFilter
        ));
        assert!(matches!(
            convert_finish_reason("function_call"),
            FinishReason::FunctionCall
        ));
        match convert_finish_reason("other") {
            FinishReason::Other(s) => assert_eq!(s, "other"),
            other => panic!("unexpected finish reason: {other:?}"),
        }

        let usage = OpenAiUsage {
            prompt_tokens: Some(1),
            completion_tokens: Some(2),
            total_tokens: Some(3),
            reasoning_tokens: Some(4),
        };
        let mapped = convert_usage(usage);
        assert_eq!(mapped.prompt_tokens, Some(1));
        assert_eq!(mapped.completion_tokens, Some(2));
        assert_eq!(mapped.total_tokens, Some(3));
        assert_eq!(mapped.reasoning_tokens, Some(4));
    }
}
