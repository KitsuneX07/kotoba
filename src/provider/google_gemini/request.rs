use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::types::{
    AudioContent, ChatRequest, ContentPart, FileContent, ImageContent, ImageSource, MediaSource,
    Message, ResponseFormat, TextContent, ToolChoice, ToolDefinition, ToolKind, VideoContent,
};

/// Builds a Google Gemini GenerateContent request body.
///
/// Unlike OpenAI Chat, Gemini encodes the model in the path:
/// `POST /v1beta/models/{model}:generateContent`. The `model` parameter only affects the URL and
/// therefore does not appear inside the JSON body.
pub(crate) fn build_gemini_body(
    request: &ChatRequest,
    _model: &str,
    _stream: bool,
) -> Result<Value, LLMError> {
    let mut body = Map::new();

    // 1. Fold system/developer roles into `system_instruction`; add the rest to `contents`.
    let mut system_texts = Vec::new();
    let mut contents = Vec::new();
    for message in &request.messages {
        match message.role.0.as_str() {
            "system" | "developer" => {
                if let Some(text) = extract_text_from_message(message) {
                    system_texts.push(text);
                }
            }
            _ => {
                contents.push(convert_message(message)?);
            }
        }
    }

    if contents.is_empty() {
        return Err(LLMError::Validation {
            message: "Gemini GenerateContent request requires at least one content message"
                .to_string(),
        });
    }
    body.insert("contents".to_string(), Value::Array(contents));

    if !system_texts.is_empty() {
        // Gemini's `system_instruction` currently expects text, so concatenate system/dev messages.
        let system_text = system_texts.join("\n\n");
        body.insert(
            "system_instruction".to_string(),
            json!({
                "role": "system",
                "parts": [ { "text": system_text } ]
            }),
        );
    }

    // 2. Sampling and generation configuration maps to `generationConfig`.
    if let Some(gen_cfg) = build_generation_config(request)? {
        body.insert("generationConfig".to_string(), gen_cfg);
    }

    // 3. Tools and `toolConfig`.
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(convert_tools(&request.tools)?),
        );
    }
    if let Some(choice) = &request.tool_choice {
        if let Some(config) = convert_tool_choice(choice)? {
            body.insert("toolConfig".to_string(), config);
        }
    }

    // 4. Pass metadata through verbatim.
    if let Some(metadata) = &request.metadata {
        let meta: Map<String, Value> = metadata.clone().into_iter().collect();
        body.insert("metadata".to_string(), Value::Object(meta));
    }

    // 5. Forward provider-specific extras (safetySettings, cachedContent, etc.).
    for (k, v) in &request.options.extra {
        body.insert(k.clone(), v.clone());
    }

    Ok(Value::Object(body))
}

/// Converts a unified [`Message`] into Gemini `Content`.
fn convert_message(message: &Message) -> Result<Value, LLMError> {
    let mut obj = Map::new();

    // Map assistant to Gemini's `model`; forward other roles as-is.
    let role = match message.role.0.as_str() {
        "assistant" => "model",
        other => other,
    };
    obj.insert("role".to_string(), Value::String(role.to_string()));

    let mut parts = Vec::new();
    for part in &message.content {
        parts.push(convert_content_part(part)?);
    }
    obj.insert("parts".to_string(), Value::Array(parts));

    Ok(Value::Object(obj))
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

/// Converts a unified [`ContentPart`] into Gemini `Part` JSON.
fn convert_content_part(part: &ContentPart) -> Result<Value, LLMError> {
    match part {
        ContentPart::Text(TextContent { text }) => Ok(json!({ "text": text })),
        ContentPart::Image(ImageContent { source, .. }) => match source {
            ImageSource::Url { url } => Ok(json!({
                "fileData": {
                    "mimeType": "application/octet-stream",
                    "fileUri": url,
                }
            })),
            ImageSource::Base64 { data, mime_type } => {
                let mime = mime_type.as_deref().unwrap_or("image/jpeg");
                Ok(json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": data,
                    }
                }))
            }
            ImageSource::FileId { file_id } => Ok(json!({
                "fileData": {
                    "mimeType": "application/octet-stream",
                    "fileUri": file_id,
                }
            })),
        },
        ContentPart::Audio(AudioContent {
            source, mime_type, ..
        }) => {
            let mime = mime_type.as_deref().unwrap_or("audio/mpeg");
            match source {
                MediaSource::Inline { data } => Ok(json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": data,
                    }
                })),
                MediaSource::Url { url } => Ok(json!({
                    "fileData": {
                        "mimeType": mime,
                        "fileUri": url,
                    }
                })),
                MediaSource::FileId { file_id } => Ok(json!({
                    "fileData": {
                        "mimeType": mime,
                        "fileUri": file_id,
                    }
                })),
            }
        }
        ContentPart::Video(VideoContent {
            source, mime_type, ..
        }) => {
            let mime = mime_type.as_deref().unwrap_or("video/mp4");
            match source {
                MediaSource::Inline { data } => Ok(json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": data,
                    }
                })),
                MediaSource::Url { url } => Ok(json!({
                    "fileData": {
                        "mimeType": mime,
                        "fileUri": url,
                    }
                })),
                MediaSource::FileId { file_id } => Ok(json!({
                    "fileData": {
                        "mimeType": mime,
                        "fileUri": file_id,
                    }
                })),
            }
        }
        ContentPart::File(FileContent { file_id, .. }) => Ok(json!({
            "fileData": {
                "mimeType": "application/octet-stream",
                "fileUri": file_id,
            }
        })),
        ContentPart::Data { data } => Ok(data.clone()),
        ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => Err(LLMError::Validation {
            message:
                "Gemini request messages do not support embedding ToolCall/ToolResult directly"
                    .to_string(),
        }),
    }
}

/// Builds the `generationConfig` field.
fn build_generation_config(request: &ChatRequest) -> Result<Option<Value>, LLMError> {
    let mut cfg: Option<Map<String, Value>> = None;

    // Helper: lazily create the generationConfig map.
    fn ensure_map(cfg: &mut Option<Map<String, Value>>) -> &mut Map<String, Value> {
        if cfg.is_none() {
            *cfg = Some(Map::new());
        }
        cfg.as_mut().unwrap()
    }

    // Sampling parameters.
    if let Some(temperature) = request.options.temperature {
        ensure_map(&mut cfg).insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.options.top_p {
        ensure_map(&mut cfg).insert("topP".to_string(), Value::from(top_p));
    }
    if let Some(max_tokens) = request.options.max_output_tokens {
        ensure_map(&mut cfg).insert("maxOutputTokens".to_string(), Value::from(max_tokens));
    }
    if let Some(penalty) = request.options.presence_penalty {
        ensure_map(&mut cfg).insert("presencePenalty".to_string(), Value::from(penalty));
    }
    if let Some(penalty) = request.options.frequency_penalty {
        ensure_map(&mut cfg).insert("frequencyPenalty".to_string(), Value::from(penalty));
    }

    // Response format mapping driven by `response_format` (JSON mode / JSON Schema).
    if let Some(format) = &request.response_format {
        match format {
            ResponseFormat::Text => {
                // `text/plain` is the default; nothing to set.
            }
            ResponseFormat::JsonObject => {
                ensure_map(&mut cfg).insert(
                    "response_mime_type".to_string(),
                    Value::String("application/json".to_string()),
                );
            }
            ResponseFormat::JsonSchema { schema } => {
                let map = ensure_map(&mut cfg);
                map.insert(
                    "response_mime_type".to_string(),
                    Value::String("application/json".to_string()),
                );
                map.insert("response_schema".to_string(), schema.clone());
            }
            ResponseFormat::Custom(value) => {
                // Treat Custom as a full generationConfig where the caller controls every field.
                return Ok(Some(value.clone()));
            }
        }
    }

    Ok(cfg.map(Value::Object))
}

/// Maps [`ToolDefinition`] values into the Gemini `tools` array.
fn convert_tools(tools: &[ToolDefinition]) -> Result<Vec<Value>, LLMError> {
    let mut result = Vec::new();
    for tool in tools {
        match &tool.kind {
            ToolKind::Function => {
                let mut decl = Map::new();
                decl.insert("name".to_string(), Value::String(tool.name.clone()));
                if let Some(desc) = &tool.description {
                    decl.insert("description".to_string(), Value::String(desc.clone()));
                }
                if let Some(schema) = &tool.input_schema {
                    decl.insert("parameters".to_string(), schema.clone());
                }
                // Each function becomes its own tool entry: `{ "functionDeclarations": [ { .. } ] }`.
                let mut tool_obj = Map::new();
                tool_obj.insert(
                    "functionDeclarations".to_string(),
                    Value::Array(vec![Value::Object(decl)]),
                );
                result.push(Value::Object(tool_obj));
            }
            ToolKind::Custom { name, config } => {
                // Custom assumes the caller already constructed a Gemini-compliant tool config.
                if let Some(cfg) = config {
                    result.push(cfg.clone());
                } else {
                    // Minimal fallback: wrap a tool with a type and name.
                    result.push(json!({ "type": name, "name": tool.name }));
                }
            }
            _ => {
                return Err(LLMError::Validation {
                    message: "Gemini tools currently only support function definitions or custom tool configs"
                        .to_string(),
                });
            }
        }
    }
    Ok(result)
}

/// ToolChoice -> toolConfig.functionCallingConfig
fn convert_tool_choice(choice: &ToolChoice) -> Result<Option<Value>, LLMError> {
    match choice {
        ToolChoice::Auto => Ok(None),
        ToolChoice::Any => Ok(Some(json!({
            "functionCallingConfig": { "mode": "any" }
        }))),
        ToolChoice::None => Ok(Some(json!({
            "functionCallingConfig": { "mode": "none" }
        }))),
        ToolChoice::Tool { name } => Ok(Some(json!({
            "functionCallingConfig": {
                "mode": "any",
                "allowedFunctionNames": [name]
            }
        }))),
        ToolChoice::Custom(value) => Ok(Some(value.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatOptions, ChatRequest, ContentPart, ImageDetail, ImageSource, Message, Role,
    };

    /// Builds the minimal text-only request body.
    #[test]
    fn build_body_with_basic_text_message() {
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
            build_gemini_body(&request, "models/gemini-2.0-flash", false).expect("body builds");

        let contents = body["contents"]
            .as_array()
            .expect("contents should be array");
        assert_eq!(contents.len(), 1);
        let msg = &contents[0];
        assert_eq!(msg["role"], json!("user"));
        let parts = msg["parts"].as_array().expect("parts should be array");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], json!({ "text": "hello" }));
    }

    /// Verifies system/developer messages fold into `system_instruction`.
    #[test]
    fn system_and_developer_fold_into_system_instruction() {
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
                        text: "Hello".to_string(),
                    })],
                    metadata: None,
                },
            ],
            options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let body =
            build_gemini_body(&request, "models/gemini-2.0-flash", false).expect("body builds");

        let system = &body["system_instruction"];
        let parts = system["parts"]
            .as_array()
            .expect("system_instruction.parts should be array");
        assert_eq!(parts.len(), 1);
        let text = parts[0]["text"].as_str().expect("text should be string");
        assert!(text.contains("You are a helpful assistant."));
        assert!(text.contains("Please answer in English."));

        let contents = body["contents"]
            .as_array()
            .expect("contents should be array");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], json!("user"));
    }

    /// Generates `generationConfig.response_schema` when JSON Schema output is requested.
    #[test]
    fn build_generation_config_with_json_schema() {
        let options = ChatOptions {
            temperature: Some(0.5),
            top_p: Some(0.9),
            max_output_tokens: Some(256),
            ..ChatOptions::default()
        };

        let schema = json!({
            "type": "OBJECT",
            "properties": {
                "name": { "type": "STRING" }
            }
        });

        let request = ChatRequest {
            messages: vec![Message {
                role: Role::user(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "hello".to_string(),
                })],
                metadata: None,
            }],
            options,
            tools: Vec::new(),
            tool_choice: None,
            response_format: Some(ResponseFormat::JsonSchema {
                schema: schema.clone(),
            }),
            metadata: None,
        };

        let body =
            build_gemini_body(&request, "models/gemini-2.0-flash", false).expect("body builds");

        let gen_cfg = body["generationConfig"]
            .as_object()
            .expect("generationConfig should be object");
        // Compare floats approximately.
        let temperature = gen_cfg["temperature"].as_f64().unwrap();
        assert!((temperature - 0.5).abs() < 1e-6);
        let top_p = gen_cfg["topP"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 1e-6);
        assert_eq!(gen_cfg["maxOutputTokens"], json!(256));
        assert_eq!(gen_cfg["response_mime_type"], json!("application/json"));
        assert_eq!(gen_cfg["response_schema"], schema);
    }

    /// Validates image content mapping for inline data and file references.
    #[test]
    fn convert_image_content_to_inline_and_file_data() {
        let img_inline = ContentPart::Image(ImageContent {
            source: ImageSource::Base64 {
                data: "b64-data".to_string(),
                mime_type: Some("image/png".to_string()),
            },
            detail: Some(ImageDetail::High),
            metadata: None,
        });
        let json_inline = convert_content_part(&img_inline).expect("inline image should convert");
        assert_eq!(
            json_inline,
            json!({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": "b64-data"
                }
            })
        );

        let img_url = ContentPart::Image(ImageContent {
            source: ImageSource::Url {
                url: "https://example.com/image.png".to_string(),
            },
            detail: None,
            metadata: None,
        });
        let json_url = convert_content_part(&img_url).expect("url image should convert");
        assert_eq!(
            json_url,
            json!({
                "fileData": {
                    "mimeType": "application/octet-stream",
                    "fileUri": "https://example.com/image.png"
                }
            })
        );
    }
}
