use std::time::Duration;

use thiserror::Error;

/// 统一的 LLM 错误类型 方便上层做降级与重试
#[derive(Debug, Error)]
pub enum LLMError {
    /// 网络或传输层失败
    #[error("transport error: {message}")]
    Transport { message: String },
    /// 鉴权失败或凭据不可用
    #[error("auth failure: {message}")]
    Auth { message: String },
    /// 触发速率限制
    #[error("rate limited: {message}")]
    RateLimit {
        /// 服务端返回的错误描述
        message: String,
        /// 告知上层退避的可选等待时长
        retry_after: Option<Duration>,
    },
    /// 请求参数不合法
    #[error("invalid request: {message}")]
    Validation { message: String },
    /// 当前供应商不支持某项能力
    #[error("feature unsupported: {feature}")]
    UnsupportedFeature { feature: &'static str },
    /// 供应商返回的特定错误
    #[error("provider {provider} error: {message}")]
    Provider {
        /// 供应商名称 例如 openai_chat
        provider: &'static str,
        /// 原始错误描述
        message: String,
    },
    /// 尚未实现但预留的能力
    #[error("not implemented: {feature}")]
    NotImplemented { feature: &'static str },
    /// 其他未知错误
    #[error("unknown error: {message}")]
    Unknown { message: String },
}

impl LLMError {
    /// 构建传输层错误的便捷方法
    pub fn transport<T: Into<String>>(message: T) -> Self {
        Self::Transport {
            message: message.into(),
        }
    }

    /// 构建供应商错误的便捷方法
    pub fn provider<T: Into<String>>(provider: &'static str, message: T) -> Self {
        Self::Provider {
            provider,
            message: message.into(),
        }
    }
}
