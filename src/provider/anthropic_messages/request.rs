use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::types::{
    ChatRequest, ContentPart, ImageContent, ImageSource, Message, ReasoningOptions, TextContent,
    ToolChoice, ToolDefinition, ToolKind, ToolResult,
};

/// 构建 Anthropic Messages 请求体
pub(crate) fn build_anthropic_body(
    request: &ChatRequest,
    model: &str,
    stream: bool,
) -> Result<Value, LLMError> {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));

    // 1. system / developer 折叠为顶层 system，其余进入 messages
    let mut system_texts = Vec::new();
    let mut messages = Vec::new();
    for message in &request.messages {
        match message.role.0.as_str() {
            "system" | "developer" => {
                if let Some(text) = extract_text_from_message(message) {
                    system_texts.push(text);
                }
            }
            _ => {
                messages.push(convert_message(message)?);
            }
        }
    }

    if messages.is_empty() {
        return Err(LLMError::Validation {
            message: "Anthropic Messages request requires at least one user/assistant message"
                .to_string(),
        });
    }
    body.insert("messages".to_string(), Value::Array(messages));

    if !system_texts.is_empty() {
        let system = system_texts.join("\n\n");
        body.insert("system".to_string(), Value::String(system));
    }

    // 2. 采样与生成控制参数
    if let Some(max_tokens) = request.options.max_output_tokens {
        body.insert("max_tokens".to_string(), Value::from(max_tokens));
    } else {
        return Err(LLMError::Validation {
            message:
                "Anthropic Messages requires ChatOptions.max_output_tokens (mapped to max_tokens)"
                    .to_string(),
        });
    }
    if let Some(temperature) = request.options.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.options.top_p {
        body.insert("top_p".to_string(), Value::from(top_p));
    }

    // 3. thinking / reasoning 配置
    if let Some(reasoning) = &request.options.reasoning {
        if let Some(obj) = build_thinking(reasoning)? {
            body.insert("thinking".to_string(), obj);
        }
    }

    // 4. tools 与 tool_choice
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(convert_tools(&request.tools)?),
        );
    }
    if let Some(choice) = &request.tool_choice {
        if let Some(value) =
            convert_tool_choice(choice, request.options.parallel_tool_calls.unwrap_or(true))?
        {
            body.insert("tool_choice".to_string(), value);
        }
    }

    // 5. metadata 直接映射
    if let Some(metadata) = &request.metadata {
        let meta: Map<String, Value> = metadata.clone().into_iter().collect();
        body.insert("metadata".to_string(), Value::Object(meta));
    }

    // 6. 透传额外 provider 配置，例如 stop_sequences 等
    for (k, v) in &request.options.extra {
        body.insert(k.clone(), v.clone());
    }

    body.insert("stream".to_string(), Value::Bool(stream));

    Ok(Value::Object(body))
}

fn extract_text_from_message(message: &Message) -> Option<String> {
    let mut buffer = String::new();
    for part in &message.content {
        if let ContentPart::Text(TextContent { text }) = part {
            if !buffer.is_empty() {
                buffer.push('\n');
            }
            buffer.push_str(text);
        }
    }
    if buffer.is_empty() {
        None
    } else {
        Some(buffer)
    }
}

fn convert_message(message: &Message) -> Result<Value, LLMError> {
    let mut obj = Map::new();

    // Anthropic 仅支持 user / assistant 角色，其它角色保守降级为 user
    let role = match message.role.0.as_str() {
        "assistant" => "assistant",
        _ => "user",
    };
    obj.insert("role".to_string(), Value::String(role.to_string()));

    let mut content_blocks = Vec::new();
    for part in &message.content {
        content_blocks.push(convert_content_part(part)?);
    }

    if content_blocks.is_empty() {
        return Err(LLMError::Validation {
            message: "message must contain at least one content part".to_string(),
        });
    }

    obj.insert("content".to_string(), Value::Array(content_blocks));
    Ok(Value::Object(obj))
}

fn convert_content_part(part: &ContentPart) -> Result<Value, LLMError> {
    match part {
        ContentPart::Text(TextContent { text }) => Ok(json!({
            "type": "text",
            "text": text
        })),
        ContentPart::Image(ImageContent { source, .. }) => match source {
            ImageSource::Base64 { data, mime_type } => {
                let media_type = mime_type.clone().unwrap_or_else(|| "image/png".to_string());
                Ok(json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data
                    }
                }))
            }
            _ => Err(LLMError::UnsupportedFeature {
                feature: "image_source_non_base64",
            }),
        },
        ContentPart::ToolResult(ToolResult {
            call_id,
            output,
            is_error,
            ..
        }) => {
            let tool_use_id = call_id.clone().ok_or_else(|| LLMError::Validation {
                message: "tool_result content requires call_id (mapped to tool_use_id)".to_string(),
            })?;
            let content_value = match output {
                Value::String(text) => Value::String(text.clone()),
                other => Value::String(other.to_string()),
            };
            Ok(json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content_value,
                "is_error": is_error
            }))
        }
        // 其它多模态或工具调用暂时不直接支持，交给上层通过 Data 自定义
        ContentPart::Audio(_)
        | ContentPart::Video(_)
        | ContentPart::File(_)
        | ContentPart::ToolCall(_) => Err(LLMError::UnsupportedFeature {
            feature: "anthropic_messages_content_type",
        }),
        ContentPart::Data { data } => Ok(data.clone()),
    }
}

fn build_thinking(reasoning: &ReasoningOptions) -> Result<Option<Value>, LLMError> {
    // 若调用方在 extra 中已经构造完整 thinking 配置，则直接透传
    if let Some(explicit) = reasoning.extra.get("thinking") {
        return Ok(Some(explicit.clone()));
    }

    if let Some(budget) = reasoning.budget_tokens {
        let mut obj = Map::new();
        obj.insert("type".to_string(), Value::String("enabled".to_string()));
        obj.insert("budget_tokens".to_string(), Value::from(budget));
        for (k, v) in &reasoning.extra {
            if k == "thinking" {
                continue;
            }
            obj.insert(k.clone(), v.clone());
        }
        Ok(Some(Value::Object(obj)))
    } else {
        Ok(None)
    }
}

fn convert_tools(tools: &[ToolDefinition]) -> Result<Vec<Value>, LLMError> {
    let mut result = Vec::new();
    for tool in tools {
        match &tool.kind {
            ToolKind::Function => {
                result.push(json!({
                    "type": "custom",
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }));
            }
            ToolKind::Custom { name, config } => {
                if let Some(cfg) = config {
                    result.push(cfg.clone());
                } else {
                    result.push(json!({ "type": name, "name": tool.name }));
                }
            }
            _ => {
                return Err(LLMError::Validation {
                    message:
                        "Anthropic tools currently only support function or custom tool configs"
                            .to_string(),
                });
            }
        }
    }
    Ok(result)
}

fn convert_tool_choice(
    choice: &ToolChoice,
    parallel_tool_calls: bool,
) -> Result<Option<Value>, LLMError> {
    let disable_parallel_tool_use = !parallel_tool_calls;
    match choice {
        ToolChoice::Auto => Ok(Some(json!({
            "type": "auto",
            "disable_parallel_tool_use": disable_parallel_tool_use
        }))),
        ToolChoice::Any => Ok(Some(json!({
            "type": "any",
            "disable_parallel_tool_use": disable_parallel_tool_use
        }))),
        ToolChoice::Tool { name } => Ok(Some(json!({
            "type": "tool",
            "name": name,
            "disable_parallel_tool_use": disable_parallel_tool_use
        }))),
        // Anthropic Messages 当前未提供显式 "none" 选项，这里选择不设置 tool_choice，
        // 调用方若希望完全禁用工具，应直接不提供 tools。
        ToolChoice::None => Ok(None),
        ToolChoice::Custom(value) => Ok(Some(value.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatOptions, ContentPart, Role};

    /// 最简文本消息请求体
    #[test]
    fn build_body_with_basic_text_message() {
        let request = ChatRequest {
            messages: vec![Message {
                role: Role::user(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "Hello, Claude".to_string(),
                })],
                metadata: None,
            }],
            options: ChatOptions {
                max_output_tokens: Some(256),
                ..ChatOptions::default()
            },
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let body =
            build_anthropic_body(&request, "claude-3-5-sonnet-20241022", false).expect("build");

        assert_eq!(body["model"], json!("claude-3-5-sonnet-20241022"));
        assert_eq!(body["max_tokens"], json!(256));
        assert_eq!(body["stream"], json!(false));

        let messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], json!("user"));
        let content = messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0],
            json!({
                "type": "text",
                "text": "Hello, Claude"
            })
        );
    }

    /// system / developer 折叠为 system
    #[test]
    fn fold_system_and_developer_into_system_field() {
        let request = ChatRequest {
            messages: vec![
                Message {
                    role: Role::system(),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "你是一个有帮助的助手。".to_string(),
                    })],
                    metadata: None,
                },
                Message {
                    role: Role("developer".to_string()),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "请用简体中文回答。".to_string(),
                    })],
                    metadata: None,
                },
                Message {
                    role: Role::user(),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "你好！".to_string(),
                    })],
                    metadata: None,
                },
            ],
            options: ChatOptions {
                max_output_tokens: Some(128),
                ..ChatOptions::default()
            },
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let body =
            build_anthropic_body(&request, "claude-3-5-sonnet-20241022", false).expect("build");

        let system = body["system"].as_str().expect("system should be string");
        assert!(system.contains("你是一个有帮助的助手。"));
        assert!(system.contains("请用简体中文回答。"));

        let messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], json!("user"));
    }
}
