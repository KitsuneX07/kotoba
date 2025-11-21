//! Shared data structures modeling multimodal chat requests and responses.
//!
//! These types normalize provider-specific payloads so the rest of the crate can stay
//! agnostic of individual API differences.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Chat role string compatible with provider-specific semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Role(pub String);

impl Role {
    pub fn system() -> Self {
        Self("system".to_string())
    }

    pub fn user() -> Self {
        Self("user".to_string())
    }

    pub fn assistant() -> Self {
        Self("assistant".to_string())
    }
}

/// Normalized chat message shared across providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role associated with this message.
    pub role: Role,
    /// Optional vendor-specific name attribute.
    pub name: Option<String>,
    /// Multimodal content parts provided in order.
    #[serde(default)]
    pub content: Vec<ContentPart>,
    /// Arbitrary metadata forwarded to providers.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Multimodal content part covering text, media, tools, and vendor data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content variant.
    Text(TextContent),
    /// Image content variant.
    Image(ImageContent),
    /// Audio content variant.
    Audio(AudioContent),
    /// Video content variant.
    Video(VideoContent),
    /// File reference variant.
    File(FileContent),
    /// Tool invocation emitted by the assistant.
    ToolCall(ToolCall),
    /// Tool execution result authored by the tool role.
    ToolResult(ToolResult),
    /// Vendor-defined or opaque content payload.
    Data { data: Value },
}

/// Textual content payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    /// Plain UTF-8 text.
    pub text: String,
}

/// Image payload compatible with OpenAI and Anthropic semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    /// Source describing where the image bytes come from.
    pub source: ImageSource,
    /// Optional detail hints such as OpenAI or Anthropic detail levels.
    pub detail: Option<ImageDetail>,
    /// Additional metadata forwarded verbatim.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Source for an image input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// Public URL accessible by the provider.
    Url { url: String },
    /// Base64-encoded inline payload.
    Base64 {
        data: String,
        mime_type: Option<String>,
    },
    /// Provider-managed file identifier.
    FileId { file_id: String },
}

/// Detail preset requested for image inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

/// Audio payload attached to a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    /// Underlying media source description.
    pub source: MediaSource,
    /// MIME type describing the audio format.
    pub mime_type: Option<String>,
    /// Custom metadata map.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Video payload attached to a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoContent {
    /// Underlying media source description.
    pub source: MediaSource,
    /// MIME type describing the video format.
    pub mime_type: Option<String>,
    /// Custom metadata map.
    pub metadata: Option<HashMap<String, Value>>,
}

/// File reference that can be resolved by providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    /// Provider file identifier.
    pub file_id: String,
    /// Optional hint describing the use case for the file.
    pub purpose: Option<String>,
    /// Custom metadata map.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Unified media source definition reused by audio and video parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaSource {
    /// base64 inline
    Inline { data: String },
    /// Provider-managed file identifier.
    FileId { file_id: String },
    /// Public URL accessible by the provider.
    Url { url: String },
}

/// Declarative definition of a tool available to the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name exposed to the model.
    pub name: String,
    /// Natural-language description of the tool purpose.
    pub description: Option<String>,
    /// Optional JSON Schema describing the input payload.
    pub input_schema: Option<Value>,
    /// Category of the tool implementation.
    pub kind: ToolKind,
    /// Provider-specific metadata forwarded untouched.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Enumerates supported tool kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolKind {
    /// Custom function definition.
    Function,
    /// File search helper.
    FileSearch,
    /// Web search helper.
    WebSearch,
    /// Computer-usage automation.
    ComputerUse,
    /// Provider-specific extension with optional configuration.
    Custom { name: String, config: Option<Value> },
}

/// Tool call emitted inside a chat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-supplied invocation identifier.
    pub id: Option<String>,
    /// Unique name exposed to the model.
    pub name: String,
    /// Structured arguments serialized as JSON.
    pub arguments: Value,
    /// Category of the tool implementation.
    pub kind: ToolCallKind,
}

/// Tool call categories for streaming deltas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallKind {
    /// Function call.
    Function,
    /// File search call.
    FileSearch,
    /// Web search helper.
    WebSearch,
    /// Computer-usage automation.
    ComputerUse,
    /// Provider-defined custom call type.
    Custom { name: String },
}

/// Result returned by a tool execution step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Correlated call identifier.
    pub call_id: Option<String>,
    /// JSON payload returned by the tool.
    pub output: Value,
    /// Indicates whether the tool reported an error.
    #[serde(default)]
    pub is_error: bool,
    /// Optional metadata such as captured stdio.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Chat request shared across all providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Ordered list of messages to send.
    pub messages: Vec<Message>,
    /// Fine-grained chat options.
    #[serde(default)]
    pub options: ChatOptions,
    /// Tool definitions available to the assistant.
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// Strategy describing how tools may be invoked.
    pub tool_choice: Option<ToolChoice>,
    /// Optional response-formatting requirements.
    pub response_format: Option<ResponseFormat>,
    /// Vendor-specific metadata forwarded untouched.
    pub metadata: Option<HashMap<String, Value>>,
}

/// Tunable chat options supported across providers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatOptions {
    /// Optional model identifier override.
    pub model: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// top_p
    pub top_p: Option<f32>,
    /// Maximum number of output tokens.
    pub max_output_tokens: Option<u32>,
    /// presence_penalty
    pub presence_penalty: Option<f32>,
    /// frequency_penalty
    pub frequency_penalty: Option<f32>,
    /// Whether providers may execute tools in parallel.
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning extensions for providers such as OpenAI or Anthropic.
    pub reasoning: Option<ReasoningOptions>,
    /// Additional provider-specific options (service tiers, safety, etc.).
    pub extra: HashMap<String, Value>,
}

/// Configuration for reasoning extensions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningOptions {
    /// Effort presets requested from the provider.
    pub effort: Option<ReasoningEffort>,
    /// Token budget for reasoning chains.
    pub budget_tokens: Option<u32>,
    /// Map of provider-specific overrides.
    pub extra: HashMap<String, Value>,
}

/// Reasoning effort presets supported by OpenAI and Anthropic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Custom(String),
}

/// Tool-choice strategies supported across providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Provider decides when to call tools.
    Auto,
    /// Provider must invoke at least one tool.
    Any,
    /// Tools are disabled for the request.
    None,
    /// Force a specific tool by name.
    Tool { name: String },
    /// Custom serialized configuration passed directly to the provider.
    Custom(Value),
}

/// Response-formatting modes supported by providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Free-form text output.
    Text,
    /// Structured JSON object.
    JsonObject,
    /// JSON Schema
    JsonSchema { schema: Value },
    /// Provider-specific response descriptor.
    Custom(Value),
}

/// Aggregated chat response returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Outputs produced by the model (messages, tools, etc.).
    pub outputs: Vec<OutputItem>,
    /// Token usage accounting.
    pub usage: Option<TokenUsage>,
    /// Why the response stopped.
    pub finish_reason: Option<FinishReason>,
    /// Effective model identifier reported by the provider.
    pub model: Option<String>,
    /// Metadata about the provider invocation.
    pub provider: ProviderMetadata,
}

/// Individual output entry emitted by the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputItem {
    /// Completed assistant message.
    Message { message: Message, index: usize },
    /// Tool invocation emitted by the assistant.
    ToolCall { call: ToolCall, index: usize },
    /// Tool execution result authored by the tool role.
    ToolResult { result: ToolResult, index: usize },
    /// Reasoning trace text.
    Reasoning { text: String, index: usize },
    /// Provider-specific payload.
    Custom { data: Value, index: usize },
}

/// Streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    /// Incremental events produced by streaming responses.
    pub events: Vec<ChatEvent>,
    /// Optional real-time token usage updates.
    pub usage: Option<TokenUsage>,
    /// Indicates whether this is the terminal chunk.
    pub is_terminal: bool,
    /// Metadata about the provider invocation.
    pub provider: ProviderMetadata,
}

/// Streaming event emitted as part of a [`ChatChunk`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatEvent {
    /// Text delta.
    MessageDelta(MessageDelta),
    /// Tool-call delta.
    ToolCallDelta(ToolCallDelta),
    /// Tool-result delta.
    ToolResultDelta(ToolResultDelta),
    /// Provider-specific raw event.
    Custom { data: Value },
}

/// Delta describing textual content generated so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Target message index within the response.
    pub index: usize,
    /// Optional role override.
    pub role: Option<Role>,
    /// Incremental content fragments.
    pub content: Vec<ContentDelta>,
    /// Why the response stopped.
    pub finish_reason: Option<FinishReason>,
}

/// Variants for streamed content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Text fragment appended to the message.
    Text { text: String },
    /// Rich JSON fragment.
    Json { value: Value },
    /// Embedded tool-call delta.
    ToolCall { delta: ToolCallDelta },
}

/// Delta describing the ongoing tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Tool call index.
    pub index: usize,
    /// Streaming identifier assigned by the provider.
    pub id: Option<String>,
    /// Unique name exposed to the model.
    pub name: Option<String>,
    /// Arguments appended so far.
    pub arguments_delta: Option<String>,
    /// Category of the tool implementation.
    pub kind: Option<ToolCallKind>,
    /// Indicates whether the call finished.
    pub is_finished: bool,
}

/// Delta describing tool results during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultDelta {
    /// Tool call index.
    pub index: usize,
    /// Provider-supplied invocation identifier.
    pub call_id: Option<String>,
    /// Output fragment added so far.
    pub output_delta: Option<String>,
    /// Indicates whether the result is an error.
    pub is_error: Option<bool>,
    /// Indicates whether the call finished.
    pub is_finished: bool,
}

/// Token usage metrics collected from the provider.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// prompt tokens
    pub prompt_tokens: Option<u64>,
    /// completion tokens
    pub completion_tokens: Option<u64>,
    /// reasoning tokens
    pub reasoning_tokens: Option<u64>,
    /// Total tokens across prompt, completion, and reasoning.
    pub total_tokens: Option<u64>,
    /// Provider-specific accounting details.
    pub details: Option<HashMap<String, Value>>,
}

/// Why a chat response stopped generating content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
    Error,
    Other(String),
}

/// Provider metadata returned with each response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderMetadata {
    /// Provider identifier such as `openai_chat`.
    pub provider: String,
    /// Upstream request identifier.
    pub request_id: Option<String>,
    /// Endpoint description or URL.
    pub endpoint: Option<String>,
    /// Raw response excerpt for debugging.
    pub raw: Option<Value>,
}

/// Capability descriptor used to filter providers at runtime.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CapabilityDescriptor {
    /// Whether the provider supports streaming outputs.
    pub supports_stream: bool,
    /// Whether image inputs are supported.
    pub supports_image_input: bool,
    /// Whether audio inputs are supported.
    pub supports_audio_input: bool,
    /// Whether video inputs are supported.
    pub supports_video_input: bool,
    /// Whether tool calls are supported.
    pub supports_tools: bool,
    /// Whether structured JSON output is available.
    pub supports_structured_output: bool,
    /// Whether parallel tool calls are supported.
    pub supports_parallel_tool_calls: bool,
}
