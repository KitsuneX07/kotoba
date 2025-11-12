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
