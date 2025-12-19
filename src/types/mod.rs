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
///
/// Messages mirror the provider-agnostic format consumed by
/// [`crate::client::LLMClient`]. Each message bundles a [`Role`], optional name, and
/// a sequence of [`ContentPart`] entries so callers can mix text, images, audio,
/// or tool directives in a single request.
///
/// # Examples
///
/// ```
/// # use kotoba_llm::types::{ContentPart, Message, Role, TextContent, ImageContent, ImageSource, ImageDetail};
/// let msg = Message {
///     role: Role::user(),
///     name: Some("alice".into()),
///     content: vec![
///         ContentPart::Text(TextContent { text: "Describe this image".into() }),
///         ContentPart::Image(ImageContent {
///             source: ImageSource::Url { url: "https://example.com/img.png".into() },
///             detail: Some(ImageDetail::High),
///             metadata: None,
///         }),
///     ],
///     metadata: None,
/// };
/// assert_eq!(msg.content.len(), 2);
/// ```
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
///
/// Providers consume these variants when converting a [`Message`] into their own
/// JSON wire format. Use [`ContentPart::ToolCall`] and [`ContentPart::ToolResult`]
/// when implementing tool handoffs.
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
///
/// Inline sources keep bytes within the JSON payload, whereas URLs and file IDs
/// hand responsibility to the provider infrastructure.
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
///
/// A `ChatRequest` mimics the shape of OpenAI/Anthropic messages while normalizing
/// multimodal content and tooling metadata. Populate `messages`, configure
/// [`ChatOptions`], and optionally attach tool definitions or response-format hints
/// before passing it to [`crate::client::LLMClient::chat`].
///
/// # Examples
///
/// ```
/// # use kotoba_llm::types::{ChatRequest, ChatOptions, ContentPart, Message, Role, TextContent};
/// let request = ChatRequest {
///     messages: vec![
///         Message {
///             role: Role::system(),
///             name: None,
///             content: vec![ContentPart::Text(TextContent { text: "You are concise.".into() })],
///             metadata: None,
///         },
///         Message {
///             role: Role::user(),
///             name: None,
///             content: vec![ContentPart::Text(TextContent { text: "Summarize Rust traits.".into() })],
///             metadata: None,
///         },
///     ],
///     options: ChatOptions { temperature: Some(0.3), ..Default::default() },
///     tools: Vec::new(),
///     tool_choice: None,
///     response_format: None,
///     metadata: None,
/// };
/// assert_eq!(request.messages.len(), 2);
/// ```
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
///
/// Every field is optional so callers can only set knobs they care about. Providers
/// ignore unknown fields or fall back to their documented defaults.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatOptions {
    /// Optional model identifier override.
    pub model: Option<String>,
    /// Sampling temperature, typically within `0.0..=2.0`.
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter where `1.0` disables the filter.
    pub top_p: Option<f32>,
    /// Maximum number of output tokens returned by the provider.
    pub max_output_tokens: Option<u32>,
    /// Encourages models to talk about new topics (`-2.0..=2.0`).
    pub presence_penalty: Option<f32>,
    /// Discourages repeating identical tokens (`-2.0..=2.0`).
    pub frequency_penalty: Option<f32>,
    /// Whether providers may execute tool calls in parallel.
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
///
/// Responses carry a list of [`OutputItem`]s (messages, tool calls, reasoning
/// traces, etc.) plus optional token usage and finish metadata. This mirrors the
/// union of OpenAI, Anthropic, and Gemini style payloads while remaining
/// provider-agnostic.
///
/// # Examples
///
/// ```
/// # use kotoba_llm::types::{ChatResponse, OutputItem, Message, Role, ContentPart, TextContent, ProviderMetadata};
/// let response = ChatResponse {
///     outputs: vec![OutputItem::Message {
///         index: 0,
///         message: Message {
///             role: Role::assistant(),
///             name: None,
///             content: vec![ContentPart::Text(TextContent { text: "Hello".into() })],
///             metadata: None,
///         },
///     }],
///     usage: None,
///     finish_reason: None,
///     model: Some("gpt-4o-mini".into()),
///     provider: ProviderMetadata { provider: "openai_chat".into(), ..Default::default() },
/// };
/// assert_eq!(response.outputs.len(), 1);
/// ```
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
///
/// The `index` mirrors upstream array indices so streaming deltas can be merged
/// deterministically.
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

/// Streaming chunk representing incremental response data.
///
/// Streaming transports emit one or more chunks until `is_terminal` becomes
/// `true`. Consumers should aggregate [`ChatEvent`] entries in order and finalize
/// once the terminal chunk arrives.
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
///
/// Providers often emit partial usage in streaming mode; consumers can merge
/// the optional values as they arrive.
///
/// # Examples
///
/// ```
/// # use kotoba_llm::types::TokenUsage;
/// let usage = TokenUsage {
///     prompt_tokens: Some(1200),
///     completion_tokens: Some(200),
///     reasoning_tokens: None,
///     total_tokens: Some(1400),
///     details: None,
/// };
/// assert_eq!(usage.total_tokens, Some(1400));
/// ```
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
///
/// Use this structure to correlate logs, surface request IDs to clients, or
/// surface endpoint information during incident triage.
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
///
/// [`crate::client::LLMClient`] exposes capability lookups so applications can
/// pick compatible providers before dispatching a request.
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

/// Groups provider families that share similar tokenization characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    /// OpenAI Chat/Responses style tokenizer with ~4 ASCII chars per token.
    OpenAI,
    /// Anthropic Claude models follow OpenAI-like heuristics for mixed locales.
    Anthropic,
    /// Google Gemini tokenizes English slightly more aggressively (~4.5 chars/token).
    GoogleGemini,
}

/// Estimates token counts using provider-specific heuristics.
///
/// The estimator favors simplicity over exact parity with vendor tokenizers, so
/// the returned counts are approximate but deterministic. Applications can use
/// it for budgeting requests, enforcing safety margins, or pre-validating
/// dynamic prompts before dispatching a network call.
#[derive(Debug, Clone)]
pub struct TokenEstimator {
    provider_type: ProviderType,
}

impl TokenEstimator {
    /// Creates an estimator tuned to a specific provider family.
    pub fn new(provider_type: ProviderType) -> Self {
        Self { provider_type }
    }

    /// Returns the configured provider type.
    pub fn provider_type(&self) -> ProviderType {
        self.provider_type
    }

    /// Estimates the number of tokens for a piece of text using provider heuristics.
    pub fn estimate_text(&self, text: &str) -> usize {
        let mut total_chars = 0usize;
        let mut ascii_chars = 0usize;

        for ch in text.chars() {
            total_chars += 1;
            if ch.is_ascii() {
                ascii_chars += 1;
            }
        }

        if total_chars == 0 {
            return 0;
        }

        let ascii_ratio = ascii_chars as f64 / total_chars as f64;
        let chars_per_token = match self.provider_type {
            ProviderType::OpenAI | ProviderType::Anthropic => 2.0 + 2.0 * ascii_ratio,
            ProviderType::GoogleGemini => 4.5,
        };

        ((total_chars as f64) / chars_per_token).ceil() as usize
    }

    /// Estimates the tokens for an entire chat request.
    ///
    /// The helper accounts for system/user/assistant roles, payload metadata,
    /// and modal content (text, audio, images, video, etc.). It returns a
    /// [`TokenEstimate`] that exposes both totals and a per-role breakdown so
    /// callers can spot oversized messages quickly.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::types::{ChatRequest, ChatOptions, ContentPart, Message, Role, TextContent, TokenEstimator, ProviderType};
    /// let request = ChatRequest {
    ///     messages: vec![
    ///         Message {
    ///             role: Role::system(),
    ///             name: None,
    ///             content: vec![ContentPart::Text(TextContent { text: "You are terse".into() })],
    ///             metadata: None,
    ///         },
    ///         Message {
    ///             role: Role::user(),
    ///             name: None,
    ///             content: vec![ContentPart::Text(TextContent { text: "Explain enums".into() })],
    ///             metadata: None,
    ///         },
    ///     ],
    ///     options: ChatOptions::default(),
    ///     tools: vec![],
    ///     tool_choice: None,
    ///     response_format: None,
    ///     metadata: None,
    /// };
    /// let estimate = TokenEstimator::new(ProviderType::OpenAI).estimate_request(&request);
    /// assert!(estimate.total > 0);
    /// assert!(estimate.by_role.contains_key("system"));
    /// ```
    pub fn estimate_request(&self, request: &ChatRequest) -> TokenEstimate {
        const OVERHEAD_PER_MESSAGE: usize = 4;

        let mut total = 0usize;
        let mut by_role: HashMap<String, usize> = HashMap::new();

        for message in &request.messages {
            let mut message_tokens = OVERHEAD_PER_MESSAGE;

            for part in &message.content {
                message_tokens += self.estimate_content_part(part, &message.role);
            }

            *by_role.entry(message.role.0.clone()).or_insert(0) += message_tokens;
            total += message_tokens;
        }

        if !request.tools.is_empty() {
            total += request.tools.len() * 50;
        }

        if let Some(format) = &request.response_format {
            total += self.estimate_response_format(format);
        }

        TokenEstimate {
            total,
            by_role,
            overhead: OVERHEAD_PER_MESSAGE * request.messages.len(),
        }
    }

    fn estimate_content_part(&self, part: &ContentPart, role: &Role) -> usize {
        match part {
            ContentPart::Text(text) => self.estimate_text(&text.text),
            ContentPart::Image(image) => self.estimate_image_tokens(image),
            ContentPart::Audio(audio) => self.estimate_audio_tokens(audio, role),
            ContentPart::Video(video) => self.estimate_media_tokens(video.metadata.as_ref()),
            ContentPart::File(file) => self.estimate_text(&file.file_id),
            ContentPart::ToolCall(call) => {
                self.estimate_text(&serde_json::to_string(call).unwrap_or_default())
            }
            ContentPart::ToolResult(result) => {
                self.estimate_text(&serde_json::to_string(result).unwrap_or_default())
            }
            ContentPart::Data { data } => self.estimate_text(&data.to_string()),
        }
    }

    fn estimate_image_tokens(&self, image: &ImageContent) -> usize {
        let base = match self.provider_type {
            ProviderType::GoogleGemini => 600,
            _ => 760,
        };

        let detail_multiplier = match image.detail {
            Some(ImageDetail::High) => 2,
            Some(ImageDetail::Low) => 1,
            _ => 1,
        };

        base * detail_multiplier
    }

    fn estimate_audio_tokens(&self, audio: &AudioContent, role: &Role) -> usize {
        let duration_ms = audio
            .metadata
            .as_ref()
            .and_then(|meta| meta.get("duration_ms").and_then(|v| v.as_f64()))
            .or_else(|| {
                audio.metadata.as_ref().and_then(|meta| {
                    meta.get("duration_seconds")
                        .and_then(|v| v.as_f64().map(|seconds| seconds * 1000.0))
                })
            });

        let duration_ms = match duration_ms {
            Some(value) if value > 0.0 => value,
            _ => return 0,
        };

        let per_token_ms = if role.0 == "assistant" { 50.0 } else { 100.0 };
        (duration_ms / per_token_ms).ceil() as usize
    }

    fn estimate_media_tokens(&self, metadata: Option<&HashMap<String, Value>>) -> usize {
        metadata
            .and_then(|meta| meta.get("duration_ms").and_then(|v| v.as_f64()))
            .map(|duration| (duration / 40.0).ceil() as usize)
            .unwrap_or(200)
    }

    fn estimate_response_format(&self, format: &ResponseFormat) -> usize {
        match format {
            ResponseFormat::JsonObject => 20,
            ResponseFormat::JsonSchema { schema } => self.estimate_text(&schema.to_string()),
            ResponseFormat::Custom(value) => self.estimate_text(&value.to_string()),
            ResponseFormat::Text => 0,
        }
    }
}

/// Token Estimate breakdown for a chat request.
#[derive(Debug, Clone)]
pub struct TokenEstimate {
    /// Estimated total tokens in the request payload.
    pub total: usize,
    /// Breakdown aggregated by chat role (system/user/assistant/etc.).
    pub by_role: HashMap<String, usize>,
    /// Per-message framing overhead used in the calculation.
    pub overhead: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_estimator_scales_with_content_length() {
        let estimator = TokenEstimator::new(ProviderType::OpenAI);
        let short_tokens = estimator.estimate_text("Hello world!");
        let verbose_tokens = estimator.estimate_text(
            "Hello world! This sentence intentionally repeats itself to emulate higher load.",
        );

        assert!(verbose_tokens >= short_tokens);
        assert!(short_tokens > 0);
    }

    #[test]
    fn request_estimator_breaks_down_roles() {
        let estimator = TokenEstimator::new(ProviderType::Anthropic);
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
                    role: Role::user(),
                    name: None,
                    content: vec![ContentPart::Text(TextContent {
                        text: "Explain Rust ownership in 2 sentences.".to_string(),
                    })],
                    metadata: None,
                },
            ],
            options: ChatOptions::default(),
            tools: vec![],
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let estimate = estimator.estimate_request(&request);
        assert!(estimate.total >= estimate.overhead);
        assert_eq!(estimate.by_role.len(), 2);
        assert!(estimate.by_role.contains_key("system"));
        assert!(estimate.by_role.contains_key("user"));
    }
}
