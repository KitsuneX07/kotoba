use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 模型配置 描述一个可调用后端
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 自定义句柄 例如 `default-openai`
    pub handle: String,
    pub provider: ProviderKind,
    pub credential: Credential,
    pub default_model: Option<String>,
    pub base_url: Option<String>,
    /// 附加设置 例如 service_tier 或 safetySettings
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

/// 供应商类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GoogleGemini,
}

/// 鉴权信息
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    /// 简单 API Key
    ApiKey {
        /// header 名称 留空时按 provider 默认
        header: Option<String>,
        /// 密钥
        key: String,
    },
    /// Bearer Token
    Bearer { token: String },
    /// Google/GCP Service Account JSON
    ServiceAccount { json: Value },
    /// 无需鉴权的本地 provider
    None,
}
