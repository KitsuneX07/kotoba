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
        // Chat Completions 历史上使用 `max_tokens`，后续新增了 `max_completion_tokens`。
        // 为了兼容当前大量 OpenAI 兼容网关实现（不少网关尚未适配 `max_completion_tokens`），
        // 这里优先使用更通用、更兼容的 `max_tokens` 字段。
        body.insert("max_tokens".to_string(), Value::from(max_tokens));
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
                    // 按 OpenAI 官方规范，这里需要拼接 data: 协议前缀，形成标准 Data URL：
                    // data:<mime>;base64,<data>
                    // 这种形式同时兼容 OpenAI 官方接口和大多数 OpenAI-兼容网关（包括 newapi）。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AudioContent, ChatOptions, ChatRequest, ContentPart, FileContent, ImageContent,
        ImageDetail, ImageSource, MediaSource, Message, ReasoningOptions, Role, TextContent,
        ToolChoice, ToolDefinition, ToolKind, VideoContent,
    };
    use serde_json::json;

    /// 构造一个只包含最简文本消息的请求体
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

        let body = build_openai_body(&request, "gpt-4.1", false).expect("body should be built");
        // 顶层字段
        assert_eq!(body["model"], json!("gpt-4.1"));
        assert_eq!(body["stream"], json!(false));

        // 消息结构
        let messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg["role"], json!("user"));
        assert_eq!(msg["name"], json!("user-1"));
        let content = msg["content"].as_array().expect("content should be array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0], json!({"type": "text", "text": "hello"}));
    }

    /// 覆盖 ChatOptions 中的大部分控制字段映射
    #[test]
    fn build_body_with_options_and_metadata() {
        let mut options = ChatOptions::default();
        options.model = Some("gpt-4.1".to_string());
        options.temperature = Some(0.3);
        options.top_p = Some(0.9);
        options.max_output_tokens = Some(256);
        options.presence_penalty = Some(0.5);
        options.frequency_penalty = Some(-0.2);
        options.parallel_tool_calls = Some(true);
        options.reasoning = Some(ReasoningOptions {
            effort: Some(ReasoningEffort::High),
            budget_tokens: Some(1024),
            extra: [("reasoning_custom".to_string(), json!("custom"))]
                .into_iter()
                .collect(),
        });
        options
            .extra
            .insert("service_tier".to_string(), json!("default"));

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("trace_id".to_string(), json!("abc123"));

        let request = ChatRequest {
            messages: vec![Message {
                role: Role::system(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "system prompt".to_string(),
                })],
                metadata: None,
            }],
            options,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: Some(metadata),
        };

        let body = build_openai_body(&request, "gpt-4.1", true).expect("body should be built");

        assert_eq!(body["model"], json!("gpt-4.1"));
        // 浮点字段使用近似比较，避免 f32/JSON 之间的精度差异
        let temperature = body["temperature"].as_f64().unwrap();
        assert!((temperature - 0.3).abs() < 1e-6);
        let top_p = body["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 1e-6);
        assert_eq!(body["max_tokens"], json!(256));
        let presence_penalty = body["presence_penalty"].as_f64().unwrap();
        assert!((presence_penalty - 0.5).abs() < 1e-6);
        let frequency_penalty = body["frequency_penalty"].as_f64().unwrap();
        assert!((frequency_penalty - (-0.2_f64)).abs() < 1e-6);
        assert_eq!(body["parallel_tool_calls"], json!(true));
        assert_eq!(body["reasoning_effort"], json!("high"));
        assert_eq!(body["max_reasoning_tokens"], json!(1024));
        assert_eq!(body["reasoning_custom"], json!("custom"));
        assert_eq!(body["service_tier"], json!("default"));
        assert_eq!(body["stream"], json!(true));

        // metadata 被打包为对象
        assert_eq!(body["metadata"]["trace_id"], json!("abc123"));
    }

    /// 多模态内容（图像/音频/视频/文件/Data）的映射
    #[test]
    fn convert_various_content_parts() {
        let parts = vec![
            ContentPart::Image(ImageContent {
                source: ImageSource::Url {
                    url: "https://example.com/image.png".to_string(),
                },
                detail: Some(ImageDetail::Low),
                metadata: None,
            }),
            ContentPart::Image(ImageContent {
                source: ImageSource::Base64 {
                    data: "AAA".to_string(),
                    mime_type: Some("image/jpeg".to_string()),
                },
                detail: Some(ImageDetail::Auto),
                metadata: None,
            }),
            ContentPart::Image(ImageContent {
                source: ImageSource::FileId {
                    file_id: "file_1".to_string(),
                },
                detail: None,
                metadata: None,
            }),
            ContentPart::Audio(AudioContent {
                source: MediaSource::Inline {
                    data: "audio-b64".to_string(),
                },
                mime_type: Some("mp3".to_string()),
                metadata: None,
            }),
            ContentPart::Video(VideoContent {
                source: MediaSource::Url {
                    url: "https://example.com/video.mp4".to_string(),
                },
                mime_type: Some("mp4".to_string()),
                metadata: None,
            }),
            ContentPart::File(FileContent {
                file_id: "file_2".to_string(),
                purpose: None,
                metadata: None,
            }),
            ContentPart::Data {
                data: json!({"custom": 1}),
            },
        ];

        let converted: Vec<Value> = parts
            .iter()
            .map(|p| convert_content_part(p).expect("convert content"))
            .collect();

        // URL 图像
        assert_eq!(
            converted[0],
            json!({
                "type": "image_url",
                "image_url": { "url": "https://example.com/image.png", "detail": "low" }
            })
        );

        // Base64 图像 → data URL
        assert_eq!(
            converted[1],
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/jpeg;base64,AAA", "detail": "auto" }
            })
        );

        // FileId 图像
        assert_eq!(
            converted[2],
            json!({
                "type": "input_image",
                "input_image": { "file_id": "file_1" }
            })
        );

        // 音频
        assert_eq!(
            converted[3],
            json!({
                "type": "input_audio",
                "input_audio": {
                    "data": "audio-b64",
                    "format": "mp3"
                }
            })
        );

        // 视频
        assert_eq!(
            converted[4],
            json!({
                "type": "input_video",
                "input_video": {
                    "source": { "url": "https://example.com/video.mp4" },
                    "format": "mp4"
                }
            })
        );

        // 文件
        assert_eq!(
            converted[5],
            json!({
                "type": "file",
                "file": { "file_id": "file_2" }
            })
        );

        // Data 原样透传
        assert_eq!(converted[6], json!({"custom": 1}));
    }

    /// 工具调用和 tool_choice 映射
    #[test]
    fn convert_tools_and_tool_choice() {
        let tools = vec![ToolDefinition {
            name: "fn1".to_string(),
            description: Some("desc".to_string()),
            input_schema: Some(json!({"type": "object"})),
            kind: ToolKind::Function,
            metadata: None,
        }];
        let tools_json = convert_tools(&tools).expect("tools should convert");
        assert_eq!(
            tools_json[0],
            json!({
                "type": "function",
                "function": {
                    "name": "fn1",
                    "description": "desc",
                    "parameters": {"type": "object"}
                }
            })
        );

        // tool_choice: auto / any / none / 指定函数 / 自定义
        assert_eq!(
            convert_tool_choice(&ToolChoice::Auto).unwrap(),
            Some(json!("auto"))
        );
        assert_eq!(
            convert_tool_choice(&ToolChoice::Any).unwrap(),
            Some(json!("required"))
        );
        assert_eq!(
            convert_tool_choice(&ToolChoice::None).unwrap(),
            Some(json!("none"))
        );
        assert_eq!(
            convert_tool_choice(&ToolChoice::Tool {
                name: "fn1".to_string()
            })
            .unwrap(),
            Some(json!({"type": "function", "function": { "name": "fn1" }}))
        );
        let custom = json!({"type": "custom"});
        assert_eq!(
            convert_tool_choice(&ToolChoice::Custom(custom.clone())).unwrap(),
            Some(custom)
        );
    }

    /// tool 角色消息与 ToolResult 的组合
    #[test]
    fn tool_message_with_single_result_is_encoded_correctly() {
        use crate::types::ToolResult;

        let tool_result = ToolResult {
            call_id: Some("call_1".to_string()),
            output: json!({"ok": true}),
            is_error: false,
            metadata: None,
        };
        let message = Message {
            role: Role("tool".to_string()),
            name: None,
            content: vec![ContentPart::ToolResult(tool_result)],
            metadata: None,
        };

        let json = convert_message(&message).expect("tool message should be convertible");
        assert_eq!(json["role"], json!("tool"));
        assert_eq!(json["tool_call_id"], json!("call_1"));
        // 非字符串输出会被序列化为字符串
        assert_eq!(json["content"], json!(r#"{"ok":true}"#));
    }

    /// 多个 ToolResult 或缺少 call_id 应该触发校验错误
    #[test]
    fn tool_message_with_invalid_results_should_fail() {
        use crate::types::ToolResult;

        // 多个结果
        let tool_result = ToolResult {
            call_id: Some("id".to_string()),
            output: json!(1),
            is_error: false,
            metadata: None,
        };
        let msg_many = Message {
            role: Role("tool".to_string()),
            name: None,
            content: vec![
                ContentPart::ToolResult(tool_result.clone()),
                ContentPart::ToolResult(tool_result.clone()),
            ],
            metadata: None,
        };
        let err = convert_message(&msg_many).unwrap_err();
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("tool role expects a single ToolResult"))
            }
            other => panic!("unexpected error: {other:?}"),
        }

        // 缺少 call_id
        let msg_missing_id = Message {
            role: Role("tool".to_string()),
            name: None,
            content: vec![ContentPart::ToolResult(ToolResult {
                call_id: None,
                output: json!(1),
                is_error: false,
                metadata: None,
            })],
            metadata: None,
        };
        let err = convert_message(&msg_missing_id).unwrap_err();
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("tool message missing call_id"))
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
