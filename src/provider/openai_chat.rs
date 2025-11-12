use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::http::{
    DynHttpTransport, HttpBodyStream, HttpRequest, HttpResponse, HttpStreamResponse,
};
use crate::provider::{ChatStream, LLMProvider};
use crate::types::{
    CapabilityDescriptor, ChatChunk, ChatEvent, ChatRequest, ChatResponse, ContentDelta,
    ContentPart, FinishReason, Message, MessageDelta, OutputItem, ProviderMetadata,
    ReasoningEffort, ResponseFormat, Role, TextContent, TokenUsage, ToolCall, ToolCallDelta,
    ToolCallKind, ToolChoice, ToolDefinition, ToolKind,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// OpenAI Chat Completions Provider
pub struct OpenAiChatProvider {
    transport: DynHttpTransport,
    base_url: String,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    default_model: Option<String>,
}

impl OpenAiChatProvider {
    /// 创建带默认 base_url 的 Provider
    pub fn new(transport: DynHttpTransport, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            organization: None,
            project: None,
            default_model: None,
        }
    }

    /// 自定义 base_url
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// 配置组织 ID
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// 配置项目 ID
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// 设置默认模型
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    fn endpoint(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        }
    }

    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        );
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Accept".to_string(), "application/json".to_string());
        if let Some(org) = &self.organization {
            headers.insert("OpenAI-Organization".to_string(), org.clone());
        }
        if let Some(project) = &self.project {
            headers.insert("OpenAI-Project".to_string(), project.clone());
        }
        headers
    }

    fn resolve_model(&self, request: &ChatRequest) -> Result<String, LLMError> {
        request
            .options
            .model
            .clone()
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| LLMError::Validation {
                message: "model is required for OpenAI Chat".to_string(),
            })
    }

    fn build_openai_body(&self, request: &ChatRequest, stream: bool) -> Result<Value, LLMError> {
        let mut body = Map::new();
        body.insert(
            "model".to_string(),
            Value::String(self.resolve_model(request)?),
        );
        body.insert(
            "messages".to_string(),
            Value::Array(self.convert_messages(&request.messages)?),
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
                Value::Array(self.convert_tools(&request.tools)?),
            );
        }
        if let Some(choice) = &request.tool_choice {
            if let Some(value) = self.convert_tool_choice(choice)? {
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

    fn convert_messages(&self, messages: &[Message]) -> Result<Vec<Value>, LLMError> {
        messages
            .iter()
            .map(|msg| self.convert_message(msg))
            .collect()
    }

    fn convert_message(&self, message: &Message) -> Result<Value, LLMError> {
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
                    tool_calls.push(self.convert_tool_call(call)?);
                }
                ContentPart::ToolResult(result) => {
                    tool_results.push(result);
                }
                _ => {
                    content_parts.push(self.convert_content_part(part)?);
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

    fn convert_content_part(&self, part: &ContentPart) -> Result<Value, LLMError> {
        match part {
            ContentPart::Text(TextContent { text }) => Ok(json!({"type": "text", "text": text})),
            ContentPart::Image(image) => {
                let detail = image
                    .detail
                    .as_ref()
                    .map(|d| format_image_detail(d).to_string())
                    .unwrap_or_else(|| "auto".to_string());
                match &image.source {
                    crate::types::ImageSource::Url { url } => Ok(json!({
                        "type": "image_url",
                        "image_url": { "url": url, "detail": detail }
                    })),
                    crate::types::ImageSource::Base64 { data, mime_type } => {
                        let mime = mime_type.as_deref().unwrap_or("application/octet-stream");
                        Ok(json!({
                            "type": "image_url",
                            "image_url": { "url": format!("data:{mime};base64,{data}"), "detail": detail }
                        }))
                    }
                    crate::types::ImageSource::FileId { file_id } => Ok(json!({
                        "type": "input_image",
                        "input_image": { "file_id": file_id }
                    })),
                }
            }
            ContentPart::Audio(audio) => Ok(json!({
                "type": "input_audio",
                "input_audio": {
                    "data": match &audio.source {
                        crate::types::MediaSource::Inline { data } => data.clone(),
                        crate::types::MediaSource::FileId { file_id } => file_id.clone(),
                        crate::types::MediaSource::Url { url } => url.clone(),
                    },
                    "format": audio.mime_type.clone().unwrap_or_else(|| "wav".to_string())
                }
            })),
            ContentPart::File(file) => Ok(json!({
                "type": "file",
                "file": { "file_id": file.file_id }
            })),
            ContentPart::Video(video) => Ok(json!({
                "type": "input_video",
                "input_video": {
                    "source": match &video.source {
                        crate::types::MediaSource::Inline { data } => json!({"data": data}),
                        crate::types::MediaSource::FileId { file_id } => json!({"file_id": file_id}),
                        crate::types::MediaSource::Url { url } => json!({"url": url}),
                    },
                    "format": video.mime_type
                }
            })),
            ContentPart::Data { data } => Ok(data.clone()),
            ContentPart::ToolCall(_) | ContentPart::ToolResult(_) => Err(LLMError::Validation {
                message: "tool content must use dedicated structs".to_string(),
            }),
        }
    }

    fn convert_tool_call(&self, call: &ToolCall) -> Result<Value, LLMError> {
        if call.kind != ToolCallKind::Function {
            return Err(LLMError::Validation {
                message: "OpenAI only supports function tool calls".to_string(),
            });
        }
        let arguments =
            serde_json::to_string(&call.arguments).map_err(|err| LLMError::Validation {
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

    fn convert_tools(&self, tools: &[ToolDefinition]) -> Result<Vec<Value>, LLMError> {
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

    fn convert_tool_choice(&self, choice: &ToolChoice) -> Result<Option<Value>, LLMError> {
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

    async fn send_request(&self, body: Value) -> Result<HttpResponse, LLMError> {
        let payload = serde_json::to_vec(&body).map_err(|err| LLMError::Validation {
            message: format!("failed to serialize request: {err}"),
        })?;
        let mut request = HttpRequest::post_json(self.endpoint(), payload);
        request.headers = self.build_headers();
        self.transport.send(request).await
    }

    async fn send_stream_request(&self, body: Value) -> Result<HttpStreamResponse, LLMError> {
        let payload = serde_json::to_vec(&body).map_err(|err| LLMError::Validation {
            message: format!("failed to serialize request: {err}"),
        })?;
        let mut request = HttpRequest::post_json(self.endpoint(), payload);
        request.headers = self.build_headers();
        self.transport.send_stream(request).await
    }

    fn ensure_success(&self, response: HttpResponse) -> Result<String, LLMError> {
        let status = response.status;
        let text = response.into_string()?;
        if (200..300).contains(&status) {
            Ok(text)
        } else {
            Err(parse_openai_error(status, &text))
        }
    }

    fn try_parse<T: DeserializeOwned>(&self, text: &str) -> Result<T, LLMError> {
        serde_json::from_str(text).map_err(|err| LLMError::Provider {
            provider: self.name(),
            message: format!("failed to parse OpenAI response: {err}"),
        })
    }

    fn map_response(&self, resp: OpenAiChatResponse) -> Result<ChatResponse, LLMError> {
        let raw = serde_json::to_value(&resp).ok();
        let mut outputs = Vec::new();
        for choice in &resp.choices {
            if let Some(message) = &choice.message {
                let (msg, tool_calls) = self.convert_response_message(message.clone())?;
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
                provider: self.name().to_string(),
                request_id: None,
                endpoint: Some(self.endpoint()),
                raw,
            },
        })
    }

    fn convert_response_message(
        &self,
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
}

#[async_trait]
impl LLMProvider for OpenAiChatProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        let body = self.build_openai_body(&request, false)?;
        let response = self.send_request(body).await?;
        let text = self.ensure_success(response)?;
        let parsed: OpenAiChatResponse = self.try_parse(&text)?;
        self.map_response(parsed)
    }

    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError> {
        let body = self.build_openai_body(&request, true)?;
        let response = self.send_stream_request(body).await?;
        if !(200..300).contains(&response.status) {
            let text = collect_stream_text(response.body, self.name()).await?;
            return Err(parse_openai_error(response.status, &text));
        }
        Ok(create_stream(response.body, self.name(), self.endpoint()))
    }

    fn capabilities(&self) -> CapabilityDescriptor {
        CapabilityDescriptor {
            supports_stream: true,
            supports_image_input: true,
            supports_audio_input: true,
            supports_video_input: false,
            supports_tools: true,
            supports_structured_output: true,
            supports_parallel_tool_calls: true,
        }
    }

    fn name(&self) -> &'static str {
        "openai_chat"
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAiChatResponse {
    id: String,
    object: String,
    created: Option<u64>,
    model: String,
    choices: Vec<OpenAiResponseChoice>,
    usage: Option<OpenAiUsage>,
    service_tier: Option<String>,
    system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAiResponseChoice {
    index: usize,
    message: Option<OpenAiResponseMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiResponseMessage {
    role: Option<String>,
    #[serde(default)]
    content: Option<OpenAiMessageContent>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallResponse>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiMessagePart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiMessagePart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<OpenAiImageUrl>,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiImageUrl {
    url: String,
    #[serde(default)]
    detail: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiToolCallResponse {
    id: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    function: Option<OpenAiToolFunction>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiToolFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiStreamChoice {
    index: usize,
    #[serde(default)]
    delta: Option<OpenAiStreamDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiStreamDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<OpenAiDeltaContent>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
enum OpenAiDeltaContent {
    Parts(Vec<OpenAiMessagePart>),
    Text(String),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    function: Option<OpenAiToolFunctionDelta>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAiToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

fn convert_usage(usage: OpenAiUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        total_tokens: usage.total_tokens,
        details: None,
    }
}

fn convert_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        "function_call" => FinishReason::FunctionCall,
        other => FinishReason::Other(other.to_string()),
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

fn create_stream(body: HttpBodyStream, provider: &'static str, endpoint: String) -> ChatStream {
    Box::pin(OpenAiSseStream::new(body, provider, endpoint))
}

struct OpenAiSseStream {
    body: HttpBodyStream,
    buffer: Vec<u8>,
    data_lines: Vec<Vec<u8>>,
    pending: VecDeque<Result<ChatChunk, LLMError>>,
    provider: &'static str,
    endpoint: String,
    stream_closed: bool,
    done_received: bool,
}

impl OpenAiSseStream {
    fn new(body: HttpBodyStream, provider: &'static str, endpoint: String) -> Self {
        Self {
            body,
            buffer: Vec::new(),
            data_lines: Vec::new(),
            pending: VecDeque::new(),
            provider,
            endpoint,
            stream_closed: false,
            done_received: false,
        }
    }

    fn handle_line(&mut self, line: Vec<u8>) {
        if line.starts_with(b"data:") {
            let mut data = line[5..].to_vec();
            if let Some(first) = data.first() {
                if *first == b' ' {
                    data.remove(0);
                }
            }
            self.data_lines.push(data);
        }
    }

    fn flush_event(&mut self) -> Result<(), LLMError> {
        if self.data_lines.is_empty() {
            return Ok(());
        }
        let mut joined = Vec::new();
        for (idx, mut segment) in self.data_lines.drain(..).enumerate() {
            if idx > 0 {
                joined.push(b'\n');
            }
            joined.append(&mut segment);
        }
        if joined.is_empty() {
            return Ok(());
        }
        let data = String::from_utf8(joined).map_err(|err| LLMError::Provider {
            provider: self.provider,
            message: format!("invalid UTF-8 in stream chunk: {err}"),
        })?;
        if data.trim() == "[DONE]" {
            self.done_received = true;
            let chunk = ChatChunk {
                events: Vec::new(),
                usage: None,
                is_terminal: true,
                provider: ProviderMetadata {
                    provider: self.provider.to_string(),
                    request_id: None,
                    endpoint: Some(self.endpoint.clone()),
                    raw: Some(json!({"event": "[DONE]"})),
                },
            };
            self.pending.push_back(Ok(chunk));
        } else {
            let chunk: OpenAiStreamChunk =
                serde_json::from_str(&data).map_err(|err| LLMError::Provider {
                    provider: self.provider,
                    message: format!("failed to parse stream chunk: {err}"),
                })?;
            let chat_chunk = convert_stream_chunk(chunk, self.provider, &self.endpoint)?;
            self.pending.push_back(Ok(chat_chunk));
        }
        Ok(())
    }
}

impl Stream for OpenAiSseStream {
    type Item = Result<ChatChunk, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(item) = this.pending.pop_front() {
            return Poll::Ready(Some(item));
        }
        if this.done_received && this.pending.is_empty() {
            return Poll::Ready(None);
        }
        loop {
            if this.stream_closed {
                if !this.buffer.is_empty() {
                    let line = this.buffer.drain(..).collect::<Vec<_>>();
                    this.handle_line(line);
                    if let Err(err) = this.flush_event() {
                        return Poll::Ready(Some(Err(err)));
                    }
                    if let Some(item) = this.pending.pop_front() {
                        return Poll::Ready(Some(item));
                    }
                }
                return Poll::Ready(None);
            }
            match this.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(chunk_result)) => match chunk_result {
                    Ok(bytes) => {
                        this.buffer.extend_from_slice(&bytes);
                        while let Some(pos) = find_newline(&this.buffer) {
                            let mut line: Vec<u8> = this.buffer.drain(..=pos).collect();
                            if line.last() == Some(&b'\n') {
                                line.pop();
                            }
                            if line.last() == Some(&b'\r') {
                                line.pop();
                            }
                            if line.is_empty() {
                                if let Err(err) = this.flush_event() {
                                    return Poll::Ready(Some(Err(err)));
                                }
                                if let Some(item) = this.pending.pop_front() {
                                    return Poll::Ready(Some(item));
                                }
                            } else {
                                this.handle_line(line);
                            }
                        }
                        if let Some(item) = this.pending.pop_front() {
                            return Poll::Ready(Some(item));
                        }
                    }
                    Err(err) => return Poll::Ready(Some(Err(err))),
                },
                Poll::Ready(None) => {
                    this.stream_closed = true;
                    if let Err(err) = this.flush_event() {
                        return Poll::Ready(Some(Err(err)));
                    }
                    return this
                        .pending
                        .pop_front()
                        .map_or(Poll::Ready(None), |item| Poll::Ready(Some(item)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn find_newline(buffer: &[u8]) -> Option<usize> {
    buffer.iter().position(|b| *b == b'\n')
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
                    role: delta.role.clone().map(Role),
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

async fn collect_stream_text(
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

fn format_reasoning_effort(effort: &ReasoningEffort) -> String {
    match effort {
        ReasoningEffort::Low => "low".to_string(),
        ReasoningEffort::Medium => "medium".to_string(),
        ReasoningEffort::High => "high".to_string(),
        ReasoningEffort::Custom(value) => value.clone(),
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

fn format_image_detail(detail: &crate::types::ImageDetail) -> &'static str {
    match detail {
        crate::types::ImageDetail::Low => "low",
        crate::types::ImageDetail::High => "high",
        crate::types::ImageDetail::Auto => "auto",
    }
}

fn parse_openai_error(status: u16, body: &str) -> LLMError {
    #[derive(Deserialize)]
    struct ErrorBody {
        error: Option<InnerError>,
    }
    #[derive(Deserialize)]
    struct InnerError {
        message: Option<String>,
        #[allow(dead_code)]
        r#type: Option<String>,
        code: Option<Value>,
    }
    if let Ok(parsed) = serde_json::from_str::<ErrorBody>(body) {
        if let Some(error) = parsed.error {
            let mut message = error.message.unwrap_or_else(|| "unknown error".to_string());
            if let Some(code) = error.code {
                message = format!("{message} ({code})");
            }
            return match status {
                401 | 403 => LLMError::Auth { message },
                429 => LLMError::RateLimit {
                    message,
                    retry_after: None,
                },
                400 => LLMError::Validation { message },
                404 => LLMError::Provider {
                    provider: "openai_chat",
                    message,
                },
                _ => LLMError::Provider {
                    provider: "openai_chat",
                    message,
                },
            };
        }
    }
    LLMError::Provider {
        provider: "openai_chat",
        message: format!("status {status}: {body}"),
    }
}
