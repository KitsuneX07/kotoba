use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiChatResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: Option<u64>,
    pub(crate) model: String,
    pub(crate) choices: Vec<OpenAiResponseChoice>,
    pub(crate) usage: Option<OpenAiUsage>,
    pub(crate) service_tier: Option<String>,
    pub(crate) system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiResponseChoice {
    pub(crate) index: usize,
    pub(crate) message: Option<OpenAiResponseMessage>,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiResponseMessage {
    pub(crate) role: Option<String>,
    #[serde(default)]
    pub(crate) content: Option<OpenAiMessageContent>,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<OpenAiToolCallResponse>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub(crate) enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiMessagePart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiMessagePart {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) image_url: Option<OpenAiImageUrl>,
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiImageUrl {
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) detail: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiToolCallResponse {
    pub(crate) id: Option<String>,
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: Option<OpenAiToolFunction>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiToolFunction {
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiUsage {
    #[serde(default)]
    pub(crate) prompt_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) completion_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiStreamChunk {
    #[serde(default)]
    pub(crate) choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    pub(crate) usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiStreamChoice {
    pub(crate) index: usize,
    #[serde(default)]
    pub(crate) delta: Option<OpenAiStreamDelta>,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiStreamDelta {
    #[serde(default)]
    pub(crate) role: Option<String>,
    #[serde(default)]
    pub(crate) content: Option<OpenAiDeltaContent>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub(crate) enum OpenAiDeltaContent {
    Parts(Vec<OpenAiMessagePart>),
    Text(String),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiToolCallDelta {
    #[serde(default)]
    pub(crate) index: Option<usize>,
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(rename = "type")]
    pub(crate) kind: Option<String>,
    #[serde(default)]
    pub(crate) function: Option<OpenAiToolFunctionDelta>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiToolFunctionDelta {
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) arguments: Option<String>,
}
