use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// GenerateContentResponse 顶层结构（非流式与流式 chunk 共用）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiGenerateContentResponse {
    /// 候选回答列表
    #[serde(default)]
    pub(crate) candidates: Vec<GeminiCandidate>,
    /// 与内容过滤相关的提示反馈
    #[serde(default, rename = "promptFeedback")]
    pub(crate) prompt_feedback: Option<Value>,
    /// token 用量元信息
    #[serde(default, rename = "usageMetadata")]
    pub(crate) usage_metadata: Option<GeminiUsageMetadata>,
    /// 实际使用的模型版本
    #[serde(default, rename = "modelVersion")]
    pub(crate) model_version: Option<String>,
    /// 本次响应的 ID
    #[serde(default, rename = "responseId")]
    pub(crate) response_id: Option<String>,
    /// 其余未映射字段，透传为 provider 元数据
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// 单个候选回答
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiCandidate {
    #[serde(default)]
    pub(crate) content: Option<GeminiContent>,
    #[serde(default, rename = "finishReason")]
    pub(crate) finish_reason: Option<String>,
    #[serde(default)]
    pub(crate) index: Option<usize>,
    /// 其它字段透传，例如 safetyRatings / citationMetadata 等
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// 候选内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiContent {
    #[serde(default)]
    pub(crate) parts: Vec<GeminiPart>,
    #[serde(default)]
    pub(crate) role: Option<String>,
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// Content.part，多模态内容单元
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiPart {
    /// 纯文本
    #[serde(default)]
    pub(crate) text: Option<String>,
    /// 内联媒体数据（base64）
    #[serde(default, rename = "inlineData", alias = "inline_data")]
    pub(crate) inline_data: Option<GeminiInlineData>,
    /// 文件引用（如 File API / GCS）
    #[serde(default, rename = "fileData", alias = "file_data")]
    pub(crate) file_data: Option<GeminiFileData>,
    /// 函数调用请求
    #[serde(default, rename = "functionCall", alias = "function_call")]
    pub(crate) function_call: Option<GeminiFunctionCall>,
    /// 函数调用响应
    #[serde(default, rename = "functionResponse", alias = "function_response")]
    pub(crate) function_response: Option<GeminiFunctionResponse>,
    /// 可执行代码
    #[serde(default, rename = "executableCode", alias = "executable_code")]
    pub(crate) executable_code: Option<GeminiExecutableCode>,
    /// 代码执行结果
    #[serde(
        default,
        rename = "codeExecutionResult",
        alias = "code_execution_result"
    )]
    pub(crate) code_execution_result: Option<GeminiCodeExecutionResult>,
    /// 未来可能扩展的其它字段，透传为 JSON
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

/// 函数调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    pub(crate) name: String,
    /// 函数参数，JSON 对象
    #[serde(default)]
    pub(crate) args: Value,
}

/// 函数调用响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionResponse {
    pub(crate) name: String,
    pub(crate) response: Value,
}

/// 可执行代码片段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GeminiExecutableCode {
    pub(crate) language: String,
    pub(crate) code: String,
}

/// 代码执行结果
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
    /// 其它明细字段（各模态 token 统计等）统一放入 details
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}
