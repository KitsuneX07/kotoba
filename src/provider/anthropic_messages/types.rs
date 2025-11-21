use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Non-streaming response payload returned by Anthropic Messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicMessageResponse {
    /// Some compatibility layers omit the `id`, so keep it optional.
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) r#type: String,
    pub(crate) model: String,
    /// Anthropic always reports the assistant role.
    #[serde(default)]
    pub(crate) role: String,
    /// Ordered list of content blocks.
    #[serde(default)]
    pub(crate) content: Vec<AnthropicContentBlock>,
    /// Stop reason provided by the API.
    #[serde(default)]
    pub(crate) stop_reason: Option<String>,
    /// Custom stop sequence if set.
    #[serde(default)]
    pub(crate) stop_sequence: Option<String>,
    /// Token usage statistics.
    #[serde(default)]
    pub(crate) usage: Option<AnthropicUsage>,
    /// Additional future fields forwarded to `ProviderMetadata::raw`.
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// Single content block (text, image, tool call/result, document, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) input: Option<Value>,
    #[serde(default, rename = "tool_use_id")]
    pub(crate) tool_use_id: Option<String>,
    /// Tool result content; keep the raw JSON because it may be a string or block array.
    #[serde(default)]
    pub(crate) content: Option<Value>,
    /// Source information for images, documents, etc.
    #[serde(default)]
    pub(crate) source: Option<AnthropicImageSource>,
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// Image source structure covering the most common fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(rename = "media_type")]
    pub(crate) media_type: String,
    pub(crate) data: String,
}

/// Usage counters returned by Anthropic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicUsage {
    #[serde(default)]
    pub(crate) input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) cache_read_input_tokens: Option<u64>,
}
