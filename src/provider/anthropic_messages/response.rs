use std::collections::HashMap;

use serde_json::json;

use crate::error::LLMError;
use crate::types::{
    ChatResponse, ContentPart, FinishReason, ImageContent, ImageSource, Message, OutputItem,
    ProviderMetadata, Role, TextContent, TokenUsage, ToolCall, ToolCallKind,
};

use super::types::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessageResponse, AnthropicUsage,
};

pub(crate) fn map_response(
    resp: AnthropicMessageResponse,
    provider: &'static str,
    endpoint: String,
) -> Result<ChatResponse, LLMError> {
    let raw = serde_json::to_value(&resp).ok();

    let mut outputs = Vec::new();
    let mut message_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match convert_content_block(block)? {
            ConvertedBlock::MessagePart(part) => message_parts.push(part),
            ConvertedBlock::ToolCall(call) => tool_calls.push(call),
            ConvertedBlock::Ignored => {}
        }
    }

    // 如果有非工具内容，则构造一条完整的 assistant 消息
    if !message_parts.is_empty() {
        outputs.push(OutputItem::Message {
            message: Message {
                role: Role::assistant(),
                name: None,
                content: message_parts,
                metadata: None,
            },
            index: 0,
        });
    }

    for call in tool_calls {
        outputs.push(OutputItem::ToolCall { call, index: 0 });
    }

    let finish_reason = resp.stop_reason.as_deref().map(convert_finish_reason);
    let usage = resp.usage.as_ref().map(convert_usage);

    Ok(ChatResponse {
        outputs,
        usage,
        finish_reason,
        model: Some(resp.model),
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: resp.id,
            endpoint: Some(endpoint),
            raw,
        },
    })
}

enum ConvertedBlock {
    MessagePart(ContentPart),
    ToolCall(ToolCall),
    Ignored,
}

fn convert_content_block(block: &AnthropicContentBlock) -> Result<ConvertedBlock, LLMError> {
    match block.kind.as_str() {
        "text" => Ok(ConvertedBlock::MessagePart(ContentPart::Text(
            TextContent {
                text: block.text.clone().unwrap_or_default(),
            },
        ))),
        "image" => {
            if let Some(source) = &block.source {
                Ok(ConvertedBlock::MessagePart(ContentPart::Image(
                    convert_image_source(source),
                )))
            } else {
                // 缺少 source 时当作原始数据透传
                let value = serde_json::to_value(block).unwrap_or_else(|_| json!({}));
                Ok(ConvertedBlock::MessagePart(ContentPart::Data {
                    data: value,
                }))
            }
        }
        "tool_use" => {
            let id = block.id.clone();
            let name = block.name.clone().unwrap_or_default();
            let input = block.input.clone().unwrap_or_else(|| json!({}));
            Ok(ConvertedBlock::ToolCall(ToolCall {
                id,
                name,
                arguments: input,
                // Anthropic 工具调用本质上也是函数调用，这里统一视为 Function
                kind: ToolCallKind::Function,
            }))
        }
        // 工具结果和文档等暂时作为 Data 透传，便于上层按需解析
        "tool_result" | "document" => {
            let value = serde_json::to_value(block).unwrap_or_else(|_| json!({}));
            Ok(ConvertedBlock::MessagePart(ContentPart::Data {
                data: value,
            }))
        }
        _ => {
            let value = serde_json::to_value(block).unwrap_or_else(|_| json!({}));
            Ok(ConvertedBlock::MessagePart(ContentPart::Data {
                data: value,
            }))
        }
    }
}

fn convert_image_source(source: &AnthropicImageSource) -> ImageContent {
    ImageContent {
        source: ImageSource::Base64 {
            data: source.data.clone(),
            mime_type: Some(source.media_type.clone()),
        },
        // Anthropic 当前未公开 detail 选项，这里保守置为 None
        detail: None,
        metadata: None,
    }
}

pub(crate) fn convert_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "stop_sequence" => FinishReason::Stop,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(crate) fn convert_usage(usage: &AnthropicUsage) -> TokenUsage {
    let mut details = HashMap::new();
    if let Some(v) = usage.cache_creation_input_tokens {
        details.insert("cache_creation_input_tokens".to_string(), json!(v));
    }
    if let Some(v) = usage.cache_read_input_tokens {
        details.insert("cache_read_input_tokens".to_string(), json!(v));
    }

    TokenUsage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        reasoning_tokens: None,
        total_tokens: usage
            .input_tokens
            .zip(usage.output_tokens)
            .map(|(i, o)| i + o),
        details: if details.is_empty() {
            None
        } else {
            Some(details)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentPart, OutputItem};

    fn sample_text_response() -> AnthropicMessageResponse {
        AnthropicMessageResponse {
            id: Some("msg_1".to_string()),
            r#type: "message".to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock {
                kind: "text".to_string(),
                text: Some("你好，我是 Claude。".to_string()),
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                source: None,
                extra: HashMap::new(),
            }],
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Some(AnthropicUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                cache_creation_input_tokens: Some(1),
                cache_read_input_tokens: Some(2),
            }),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn map_basic_text_response() {
        let resp = sample_text_response();
        let mapped = map_response(
            resp,
            "anthropic_messages",
            "https://api.fake/v1/messages".into(),
        )
        .expect("map_response should succeed");

        assert_eq!(mapped.model.as_deref(), Some("claude-3-5-sonnet-20241022"));
        assert!(matches!(mapped.finish_reason, Some(FinishReason::Stop)));
        assert_eq!(mapped.provider.provider, "anthropic_messages");
        assert_eq!(
            mapped.provider.endpoint.as_deref(),
            Some("https://api.fake/v1/messages")
        );

        assert_eq!(mapped.outputs.len(), 1);
        match &mapped.outputs[0] {
            OutputItem::Message { message, index } => {
                assert_eq!(*index, 0);
                assert_eq!(message.role.0, "assistant");
                assert_eq!(message.content.len(), 1);
                match &message.content[0] {
                    ContentPart::Text(TextContent { text }) => {
                        assert_eq!(text, "你好，我是 Claude。");
                    }
                    other => panic!("unexpected content part: {other:?}"),
                }
            }
            other => panic!("unexpected output item: {other:?}"),
        }

        let usage = mapped.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
        let details = usage.details.expect("details should exist");
        assert_eq!(details["cache_creation_input_tokens"], json!(1));
        assert_eq!(details["cache_read_input_tokens"], json!(2));
    }

    #[test]
    fn map_tool_use_block_to_tool_call_output() {
        let resp = AnthropicMessageResponse {
            id: Some("msg_2".to_string()),
            r#type: "message".to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock {
                kind: "tool_use".to_string(),
                text: None,
                id: Some("toolu_1".to_string()),
                name: Some("get_weather".to_string()),
                input: Some(json!({ "location": "北京" })),
                tool_use_id: None,
                content: None,
                source: None,
                extra: HashMap::new(),
            }],
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: None,
            extra: HashMap::new(),
        };

        let mapped =
            map_response(resp, "anthropic_messages", "endpoint".into()).expect("map succeeds");

        assert_eq!(mapped.outputs.len(), 1);
        match &mapped.outputs[0] {
            OutputItem::ToolCall { call, index } => {
                assert_eq!(*index, 0);
                assert_eq!(call.id.as_deref(), Some("toolu_1"));
                assert_eq!(call.name, "get_weather");
                assert_eq!(call.arguments["location"], json!("北京"));
                assert!(matches!(call.kind, ToolCallKind::Function));
            }
            other => panic!("unexpected output item: {other:?}"),
        }
        assert!(matches!(
            mapped.finish_reason,
            Some(FinishReason::ToolCalls)
        ));
    }
}
