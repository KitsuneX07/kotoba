use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::types::{
    AudioContent, ChatRequest, ContentPart, FileContent, ImageContent, ImageDetail, ImageSource,
    MediaSource, Message, ReasoningEffort, ResponseFormat, TextContent, ToolChoice, ToolDefinition,
    ToolKind, VideoContent,
};

/// Builds the request body expected by the OpenAI Responses API.
pub(crate) fn build_openai_responses_body(
    request: &ChatRequest,
    model: &str,
    stream: bool,
) -> Result<Value, LLMError> {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));

    // Fold system/developer messages into `instructions` and treat the rest as `input`.
    let mut instructions_parts = Vec::new();
    let mut input_messages = Vec::new();
    for message in &request.messages {
        match message.role.0.as_str() {
            "system" | "developer" => {
                if let Some(text) = extract_text_from_message(message) {
                    instructions_parts.push(text);
                }
            }
            _ => {
                input_messages.push(convert_input_message(message)?);
            }
        }
    }

    if !input_messages.is_empty() {
        body.insert("input".to_string(), Value::Array(input_messages));
    }

    if !instructions_parts.is_empty() {
        let instructions = instructions_parts.join("\n\n");
        body.insert("instructions".to_string(), Value::String(instructions));
    }

    // Sampling and control parameters.
    if let Some(temperature) = request.options.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.options.top_p {
        body.insert("top_p".to_string(), Value::from(top_p));
    }
    if let Some(max_tokens) = request.options.max_output_tokens {
        body.insert("max_output_tokens".to_string(), Value::from(max_tokens));
    }
    if let Some(parallel) = request.options.parallel_tool_calls {
        body.insert("parallel_tool_calls".to_string(), Value::from(parallel));
    }

    // Map reasoning options; Responses officially documents effort plus a few extras.
    if let Some(reasoning) = &request.options.reasoning {
        let mut reasoning_obj = Map::new();
        if let Some(effort) = &reasoning.effort {
            reasoning_obj.insert(
                "effort".to_string(),
                Value::String(format_reasoning_effort(effort)),
            );
        }
        // Forward remaining fields verbatim so callers can control them.
        for (k, v) in &reasoning.extra {
            reasoning_obj.insert(k.clone(), v.clone());
        }
        if !reasoning_obj.is_empty() {
            body.insert("reasoning".to_string(), Value::Object(reasoning_obj));
        }
    }

    // tools & tool_choice
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

    // Response format maps to `text.format`.
    if let Some(format) = &request.response_format {
        // If callers already set `text` via `extra`, honor that configuration first.
        if !body.contains_key("text") {
            body.insert("text".to_string(), convert_text_config(format));
        }
    }

    // Metadata is mapped directly.
    if let Some(metadata) = &request.metadata {
        let meta: Map<String, Value> = metadata.clone().into_iter().collect();
        body.insert("metadata".to_string(), Value::Object(meta));
    }

    // Extra provider settings (include, service_tier, user, previous_response_id, etc.).
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

fn convert_input_message(message: &Message) -> Result<Value, LLMError> {
    let mut obj = Map::new();
    obj.insert("type".to_string(), Value::String("message".to_string()));
    obj.insert("role".to_string(), Value::String(message.role.0.clone()));

    let mut content_items = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => {
                return Err(LLMError::Validation {
                    message: "tool contents are not allowed in Responses input messages"
                        .to_string(),
                });
            }
            _ => {
                content_items.push(convert_content_part(part)?);
            }
        }
    }

    // Responses accepts strings or arrays for content; always use arrays to support multimodal input.
    obj.insert("content".to_string(), Value::Array(content_items));

    Ok(Value::Object(obj))
}

fn convert_content_part(part: &ContentPart) -> Result<Value, LLMError> {
    match part {
        ContentPart::Text(TextContent { text }) => {
            Ok(json!({ "type": "input_text", "text": text }))
        }
        ContentPart::Image(ImageContent { source, detail, .. }) => {
            let detail = detail.as_ref().map(format_image_detail).unwrap_or("auto");
            match source {
                ImageSource::Url { url } => Ok(json!({
                    "type": "input_image",
                    "image_url": url,
                    "detail": detail
                })),
                ImageSource::Base64 { data, mime_type } => {
                    // Match the Chat API by constructing `data:` URLs for inline content.
                    let mime = mime_type.as_deref().unwrap_or("application/octet-stream");
                    Ok(json!({
                        "type": "input_image",
                        "image_url": format!("data:{mime};base64,{data}"),
                        "detail": detail
                    }))
                }
                ImageSource::FileId { file_id } => Ok(json!({
                    "type": "input_image",
                    "file_id": file_id,
                    "detail": detail
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
        ContentPart::File(FileContent { file_id, .. }) => Ok(json!({
            "type": "input_file",
            "file_id": file_id
        })),
        ContentPart::Data { data } => Ok(data.clone()),
        ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => Err(LLMError::Validation {
            message: "tool content must use dedicated structs".to_string(),
        }),
    }
}

fn convert_tools(tools: &[ToolDefinition]) -> Result<Vec<Value>, LLMError> {
    tools
        .iter()
        .map(|tool| match &tool.kind {
            ToolKind::Function => {
                let mut obj = Map::new();
                obj.insert("type".to_string(), Value::String("function".to_string()));
                obj.insert("name".to_string(), Value::String(tool.name.clone()));
                if let Some(description) = &tool.description {
                    obj.insert(
                        "description".to_string(),
                        Value::String(description.clone()),
                    );
                }
                if let Some(schema) = &tool.input_schema {
                    obj.insert("parameters".to_string(), schema.clone());
                }
                // Responses sets function tools to strict=true by default; metadata can override when necessary.
                obj.insert("strict".to_string(), Value::Bool(true));
                if let Some(meta) = &tool.metadata {
                    for (k, v) in meta {
                        if k == "type" || k == "name" || k == "parameters" {
                            continue;
                        }
                        obj.insert(k.clone(), v.clone());
                    }
                }
                Ok(Value::Object(obj))
            }
            ToolKind::FileSearch | ToolKind::WebSearch | ToolKind::ComputerUse => {
                let mut obj: Map<String, Value> = tool
                    .metadata
                    .clone()
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                let default_type = match tool.kind {
                    ToolKind::FileSearch => "file_search",
                    ToolKind::WebSearch => "web_search_preview",
                    ToolKind::ComputerUse => "computer_use_preview",
                    ToolKind::Function | ToolKind::Custom { .. } => unreachable!(),
                };
                obj.entry("type".to_string())
                    .or_insert_with(|| Value::String(default_type.to_string()));
                Ok(Value::Object(obj))
            }
            ToolKind::Custom { name, config } => {
                if let Some(config) = config {
                    Ok(config.clone())
                } else {
                    Ok(json!({ "type": name, "name": tool.name }))
                }
            }
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

fn convert_text_config(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({
            "format": { "type": "text" }
        }),
        ResponseFormat::JsonObject => json!({
            "format": { "type": "json_object" }
        }),
        ResponseFormat::JsonSchema { schema } => json!({
            "format": {
                "type": "json_schema",
                // Use a fixed name for simplicity; callers can fully customize via the Custom variant.
                "name": "response",
                "schema": schema
            }
        }),
        // Treat Custom as the full `text` object so callers can set `format` or extra fields.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatOptions, ChatRequest, ContentPart, ImageContent, ImageDetail, ImageSource, Message,
        ReasoningOptions, Role,
    };

    /// Builds the minimal payload containing a single user text message.
    #[test]
    fn build_body_with_basic_text_input() {
        let request = ChatRequest {
            messages: vec![Message {
                role: Role::user(),
                name: Some("user-1".to_string()),
                content: vec![ContentPart::Text(TextContent {
                    text: "hello".to_string(),
                })],
                metadata: None,
            }],
            options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let body =
            build_openai_responses_body(&request, "gpt-4.1", false).expect("body should be built");

        assert_eq!(body["model"], json!("gpt-4.1"));
        assert_eq!(body["stream"], json!(false));

        let input = body["input"].as_array().expect("input should be array");
        assert_eq!(input.len(), 1);
        let msg = &input[0];
        assert_eq!(msg["type"], json!("message"));
        assert_eq!(msg["role"], json!("user"));
        let content = msg["content"].as_array().expect("content should be array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0], json!({ "type": "input_text", "text": "hello" }));
    }

    /// system / developer are folded into `instructions`, the rest go into `input`
    #[test]
    fn build_body_with_instructions_and_input_messages() {
        let mut options = ChatOptions {
            model: Some("gpt-4.1".to_string()),
            temperature: Some(0.3),
            top_p: Some(0.9),
            max_output_tokens: Some(256),
            parallel_tool_calls: Some(true),
            reasoning: Some(ReasoningOptions {
                effort: Some(ReasoningEffort::High),
                budget_tokens: None,
                extra: [("reasoning_custom".to_string(), json!("custom"))]
                    .into_iter()
                    .collect(),
            }),
            ..ChatOptions::default()
        };
        options
            .extra
            .insert("service_tier".to_string(), json!("default"));

        let request = ChatRequest {
            messages: vec![
                Message {
                    role: Role::system(),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "You are a helpful assistant.".to_string(),
                    })],
                    metadata: None,
                },
                Message {
                    role: Role("developer".to_string()),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "Please answer in English.".to_string(),
                    })],
                    metadata: None,
                },
                Message {
                    role: Role::user(),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "Hello!".to_string(),
                    })],
                    metadata: None,
                },
            ],
            options,
            tools: Vec::new(),
            tool_choice: None,
            response_format: Some(ResponseFormat::Text),
            metadata: None,
        };

        let body =
            build_openai_responses_body(&request, "gpt-4.1", true).expect("body should be built");

        assert_eq!(body["model"], json!("gpt-4.1"));
        // Compare floating-point values approximately.
        let temperature = body["temperature"].as_f64().unwrap();
        assert!((temperature - 0.3).abs() < 1e-6);
        let top_p = body["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 1e-6);
        assert_eq!(body["max_output_tokens"], json!(256));
        assert_eq!(body["parallel_tool_calls"], json!(true));
        assert_eq!(body["service_tier"], json!("default"));

        // reasoning
        let reasoning = body["reasoning"]
            .as_object()
            .expect("reasoning should be object");
        assert_eq!(reasoning["effort"], json!("high"));
        assert_eq!(reasoning["reasoning_custom"], json!("custom"));

        // `instructions` contains the folded system and developer text.
        let instructions = body["instructions"]
            .as_str()
            .expect("instructions should be string");
        assert!(instructions.contains("You are a helpful assistant."));
        assert!(instructions.contains("Please answer in English."));

        // `input` contains user messages only.
        let input = body["input"].as_array().expect("input should be array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], json!("user"));

        // text.format
        let text_cfg = body["text"].as_object().expect("text should be object");
        assert_eq!(text_cfg["format"]["type"], json!("text"));
    }

    /// Verifies that image input is mapped to `input_image` entries.
    #[test]
    fn convert_image_content_to_input_image() {
        let request = ChatRequest {
            messages: vec![Message {
                role: Role::user(),
                name: None,
                content: vec![ContentPart::Image(ImageContent {
                    source: ImageSource::Url {
                        url: "https://example.com/image.png".to_string(),
                    },
                    detail: Some(ImageDetail::High),
                    metadata: None,
                })],
                metadata: None,
            }],
            options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let body =
            build_openai_responses_body(&request, "gpt-4.1", false).expect("body should be built");
        let input = body["input"].as_array().expect("input should be array");
        let msg = &input[0];
        let content = msg["content"].as_array().expect("content should be array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0],
            json!({
                "type": "input_image",
                "image_url": "https://example.com/image.png",
                "detail": "high"
            })
        );
    }
}
