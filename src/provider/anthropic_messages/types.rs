use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Anthropic Messages 非流式响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicMessageResponse {
    /// 一些兼容层可能不会返回 id，这里使用 Option 兼容
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) r#type: String,
    pub(crate) model: String,
    /// Anthropic 始终返回 assistant 角色
    #[serde(default)]
    pub(crate) role: String,
    /// 内容块列表
    #[serde(default)]
    pub(crate) content: Vec<AnthropicContentBlock>,
    /// 停止原因
    #[serde(default)]
    pub(crate) stop_reason: Option<String>,
    /// 自定义停止序列
    #[serde(default)]
    pub(crate) stop_sequence: Option<String>,
    /// token 用量
    #[serde(default)]
    pub(crate) usage: Option<AnthropicUsage>,
    /// 未来可能扩展的字段，直接透传给 ProviderMetadata.raw
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// 单个内容块（文本 / 图像 / 工具调用 / 工具结果 / 文档等）
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
    /// 工具结果 content，可能是字符串或内容块数组，这里先保持为原始 JSON
    #[serde(default)]
    pub(crate) content: Option<Value>,
    /// 图像 / 文档等的 source
    #[serde(default)]
    pub(crate) source: Option<AnthropicImageSource>,
    #[serde(flatten)]
    pub(crate) extra: HashMap<String, Value>,
}

/// 图像 source 结构，仅覆盖最常用字段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(rename = "media_type")]
    pub(crate) media_type: String,
    pub(crate) data: String,
}

/// Usage 统计
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
