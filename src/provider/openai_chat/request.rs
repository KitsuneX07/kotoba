use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::types::{
    AudioContent, ChatRequest, ContentPart, FileContent, ImageContent, ImageDetail, ImageSource,
    MediaSource, Message, ReasoningEffort, ResponseFormat, TextContent, ToolCall, ToolCallKind,
    ToolChoice, ToolDefinition, ToolKind, VideoContent,
};

pub(crate) fn build_openai_body(
    request: &ChatRequest,
    model: &str,
    stream: bool,
) -> Result<Value, LLMError> {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert(
        "messages".to_string(),
        Value::Array(convert_messages(&request.messages)?),
    );
    if let Some(temperature) = request.options.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.options.top_p {
        body.insert("top_p".to_string(), Value::from(top_p));
    }
    if let Some(max_tokens) = request.options.max_output_tokens {
        body.insert("max_completion_tokens".to_string(), Value::from(max_tokens));
    }
    if let Some(penalty) = request.options.presence_penalty {
        body.insert("presence_penalty".to_string(), Value::from(penalty));
    }
    if let Some(penalty) = request.options.frequency_penalty {
        body.insert("frequency_penalty".to_string(), Value::from(penalty));
    }
    if let Some(parallel) = request.options.parallel_tool_calls {
        body.insert("parallel_tool_calls".to_string(), Value::from(parallel));
    }
    if let Some(reasoning) = &request.options.reasoning {
        if let Some(effort) = &reasoning.effort {
            body.insert(
                "reasoning_effort".to_string(),
                Value::String(format_reasoning_effort(effort)),
            );
        }
        if let Some(budget) = reasoning.budget_tokens {
            body.insert("max_reasoning_tokens".to_string(), Value::from(budget));
        }
        for (k, v) in &reasoning.extra {
            body.insert(k.clone(), v.clone());
        }
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(convert_tools(&request.tools)?),
        );
    }
    if let Some(choice) = &request.tool_choice {
        if let Some(value) = convert_tool_choice(choice)? {
            body.insert("tool_choice".to_string(), value);
        }
    }
    if let Some(format) = &request.response_format {
        body.insert(
            "response_format".to_string(),
            convert_response_format(format),
        );
    }
    if let Some(metadata) = &request.metadata {
        let meta: Map<String, Value> = metadata.clone().into_iter().collect();
        body.insert("metadata".to_string(), Value::Object(meta));
    }
    for (k, v) in &request.options.extra {
        body.insert(k.clone(), v.clone());
    }
    body.insert("stream".to_string(), Value::Bool(stream));
    Ok(Value::Object(body))
}

fn convert_messages(messages: &[Message]) -> Result<Vec<Value>, LLMError> {
    messages.iter().map(convert_message).collect()
}

fn convert_message(message: &Message) -> Result<Value, LLMError> {
    let mut obj = Map::new();
    obj.insert("role".to_string(), Value::String(message.role.0.clone()));
    if let Some(name) = &message.name {
        obj.insert("name".to_string(), Value::String(name.clone()));
    }

    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for part in &message.content {
        match part {
            ContentPart::ToolCall(call) => {
                tool_calls.push(convert_tool_call(call)?);
            }
            ContentPart::ToolResult(result) => {
                tool_results.push(result);
            }
            _ => {
                content_parts.push(convert_content_part(part)?);
            }
        }
    }

    if message.role.0 == "tool" {
        if tool_results.len() > 1 {
            return Err(LLMError::Validation {
                message: "tool role expects a single ToolResult content".to_string(),
            });
        }
        if let Some(result) = tool_results.first() {
            let content_string = match &result.output {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            obj.insert(
                "tool_call_id".to_string(),
                Value::String(result.call_id.clone().ok_or_else(|| LLMError::Validation {
                    message: "tool message missing call_id".to_string(),
                })?),
            );
            obj.insert("content".to_string(), Value::String(content_string));
        } else {
            obj.insert("content".to_string(), Value::Null);
        }
    } else {
        obj.insert(
            "content".to_string(),
            if content_parts.is_empty() {
                Value::Null
            } else {
                Value::Array(content_parts)
            },
        );
        if !tool_calls.is_empty() {
            obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }
    }

    Ok(Value::Object(obj))
}

fn convert_content_part(part: &ContentPart) -> Result<Value, LLMError> {
    match part {
        ContentPart::Text(TextContent { text }) => Ok(json!({"type": "text", "text": text})),
        ContentPart::Image(ImageContent { source, detail, .. }) => {
            let detail = detail.as_ref().map(format_image_detail).unwrap_or("auto");
            match source {
                ImageSource::Url { url } => Ok(json!({
                    "type": "image_url",
                    "image_url": { "url": url, "detail": detail }
                })),
                ImageSource::Base64 { data, mime_type } => {
                    let mime = mime_type.as_deref().unwrap_or("application/octet-stream");
                    Ok(json!({
                        "type": "image_url",
                        "image_url": { "url": format!("data:{mime};base64,{data}"), "detail": detail }
                    }))
                }
                ImageSource::FileId { file_id } => Ok(json!({
                    "type": "input_image",
                    "input_image": { "file_id": file_id }
                })),
            }
        }
        ContentPart::Audio(AudioContent {
            source, mime_type, ..
        }) => Ok(json!({
            "type": "input_audio",
            "input_audio": {
                "data": match source {
                    MediaSource::Inline { data } => data.clone(),
                    MediaSource::FileId { file_id } => file_id.clone(),
                    MediaSource::Url { url } => url.clone(),
                },
                "format": mime_type.clone().unwrap_or_else(|| "wav".to_string())
            }
        })),
        ContentPart::File(FileContent { file_id, .. }) => Ok(json!({
            "type": "file",
            "file": { "file_id": file_id }
        })),
        ContentPart::Video(VideoContent {
            source, mime_type, ..
        }) => Ok(json!({
            "type": "input_video",
            "input_video": {
                "source": match source {
                    MediaSource::Inline { data } => json!({"data": data}),
                    MediaSource::FileId { file_id } => json!({"file_id": file_id}),
                    MediaSource::Url { url } => json!({"url": url}),
                },
                "format": mime_type
            }
        })),
        ContentPart::Data { data } => Ok(data.clone()),
        ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => Err(LLMError::Validation {
            message: "tool content must use dedicated structs".to_string(),
        }),
    }
}

fn convert_tool_call(call: &ToolCall) -> Result<Value, LLMError> {
    if call.kind != ToolCallKind::Function {
        return Err(LLMError::Validation {
            message: "OpenAI only supports function tool calls".to_string(),
        });
    }
    let arguments = serde_json::to_string(&call.arguments).map_err(|err| LLMError::Validation {
        message: format!("invalid tool arguments: {err}"),
    })?;
    let mut obj = Map::new();
    if let Some(id) = &call.id {
        obj.insert("id".to_string(), Value::String(id.clone()));
    }
    obj.insert("type".to_string(), Value::String("function".to_string()));
    obj.insert(
        "function".to_string(),
        json!({
            "name": call.name,
            "arguments": arguments
        }),
    );
    Ok(Value::Object(obj))
}

fn convert_tools(tools: &[ToolDefinition]) -> Result<Vec<Value>, LLMError> {
    tools
        .iter()
        .map(|tool| match tool.kind {
            ToolKind::Function => Ok(json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })),
            _ => Err(LLMError::Validation {
                message: "OpenAI Chat tools only support function definitions".to_string(),
            }),
        })
        .collect()
}

fn convert_tool_choice(choice: &ToolChoice) -> Result<Option<Value>, LLMError> {
    match choice {
        ToolChoice::Auto => Ok(Some(Value::String("auto".to_string()))),
        ToolChoice::Any => Ok(Some(Value::String("required".to_string()))),
        ToolChoice::None => Ok(Some(Value::String("none".to_string()))),
        ToolChoice::Tool { name } => Ok(Some(json!({
            "type": "function",
            "function": { "name": name }
        }))),
        ToolChoice::Custom(value) => Ok(Some(value.clone())),
    }
}

fn convert_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { schema } => {
            json!({ "type": "json_schema", "json_schema": schema })
        }
        ResponseFormat::Custom(value) => value.clone(),
    }
}

fn format_reasoning_effort(effort: &ReasoningEffort) -> String {
    match effort {
        ReasoningEffort::Low => "low".to_string(),
        ReasoningEffort::Medium => "medium".to_string(),
        ReasoningEffort::High => "high".to_string(),
        ReasoningEffort::Custom(value) => value.clone(),
    }
}

fn format_image_detail(detail: &ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Auto => "auto",
    }
}
