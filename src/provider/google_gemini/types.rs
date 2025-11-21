use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Top-level GenerateContentResponse shared by sync calls and streaming chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiGenerateContentResponse {
    /// Candidate responses produced by Gemini.
    #[serde(default)]
    pub(crate) candidates: Vec<GeminiCandidate>,
    /// Prompt feedback about content filtering.
    #[serde(default, rename = "promptFeedback")]
    pub(crate) prompt_feedback: Option<Value>,
    /// Token usage metadata.
    #[serde(default, rename = "usageMetadata")]
    pub(crate) usage_metadata: Option<GeminiUsageMetadata>,
    /// Model version actually used for the response.
    #[serde(default, rename = "modelVersion")]
    pub(crate) model_version: Option<String>,
    /// Unique response identifier.
    #[serde(default, rename = "responseId")]
    pub(crate) response_id: Option<String>,
    /// Any unmapped fields forwarded to provider metadata.
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// Single candidate response entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiCandidate {
    #[serde(default)]
    pub(crate) content: Option<GeminiContent>,
    #[serde(default, rename = "finishReason")]
    pub(crate) finish_reason: Option<String>,
    #[serde(default)]
    pub(crate) index: Option<usize>,
    /// Additional fields (safety ratings, citation metadata, etc.) are forwarded.
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// Candidate content payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiContent {
    #[serde(default)]
    pub(crate) parts: Vec<GeminiPart>,
    #[serde(default)]
    pub(crate) role: Option<String>,
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// Multimodal content part emitted by Gemini.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiPart {
    /// Plain text part.
    #[serde(default)]
    pub(crate) text: Option<String>,
    /// Inline media data encoded as base64.
    #[serde(default, rename = "inlineData", alias = "inline_data")]
    pub(crate) inline_data: Option<GeminiInlineData>,
    /// File references (File API, GCS, etc.).
    #[serde(default, rename = "fileData", alias = "file_data")]
    pub(crate) file_data: Option<GeminiFileData>,
    /// Function-call request part.
    #[serde(default, rename = "functionCall", alias = "function_call")]
    pub(crate) function_call: Option<GeminiFunctionCall>,
    /// Function-call response part.
    #[serde(default, rename = "functionResponse", alias = "function_response")]
    pub(crate) function_response: Option<GeminiFunctionResponse>,
    /// Executable code snippet.
    #[serde(default, rename = "executableCode", alias = "executable_code")]
    pub(crate) executable_code: Option<GeminiExecutableCode>,
    /// Code execution result.
    #[serde(
        default,
        rename = "codeExecutionResult",
        alias = "code_execution_result"
    )]
    pub(crate) code_execution_result: Option<GeminiCodeExecutionResult>,
    /// Future extension fields forwarded as JSON.
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// InlineData
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiInlineData {
    #[serde(rename = "mimeType", alias = "mime_type")]
    pub(crate) mime_type: String,
    pub(crate) data: String,
}

/// FileData
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFileData {
    #[serde(rename = "mimeType", alias = "mime_type")]
    pub(crate) mime_type: Option<String>,
    #[serde(rename = "fileUri", alias = "file_uri")]
    pub(crate) file_uri: String,
}

/// Function-call description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    pub(crate) name: String,
    /// Function arguments represented as JSON.
    #[serde(default)]
    pub(crate) args: Value,
}

/// Function-call response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionResponse {
    pub(crate) name: String,
    pub(crate) response: Value,
}

/// Executable code snippet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiExecutableCode {
    pub(crate) language: String,
    pub(crate) code: String,
}

/// Result of executing code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiCodeExecutionResult {
    pub(crate) outcome: String,
    #[serde(default)]
    pub(crate) output: Option<String>,
}

/// UsageMetadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    pub(crate) prompt_token_count: Option<u64>,
    #[serde(rename = "cachedContentTokenCount", default)]
    pub(crate) cached_content_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount", default)]
    pub(crate) candidates_token_count: Option<u64>,
    #[serde(rename = "totalTokenCount", default)]
    pub(crate) total_token_count: Option<u64>,
    #[serde(rename = "toolUsePromptTokenCount", default)]
    pub(crate) tool_use_prompt_token_count: Option<u64>,
    #[serde(rename = "thoughtsTokenCount", default)]
    pub(crate) thoughts_token_count: Option<u64>,
    /// Additional modal-specific counts are flattened into `extra`.
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}
