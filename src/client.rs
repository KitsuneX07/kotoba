use std::collections::HashMap;

use crate::error::LLMError;
use crate::provider::{ChatStream, DynProvider};
use crate::types::{ChatRequest, ChatResponse};

/// LLM 调用入口 负责维护已注册的 Provider
pub struct LLMClient {
    providers: HashMap<String, DynProvider>,
}

impl LLMClient {
    /// 创建 Builder 便于后续注册 Provider
    pub fn builder() -> LLMClientBuilder {
        LLMClientBuilder {
            providers: HashMap::new(),
        }
    }

    /// 发送同步聊天请求
    pub async fn chat(&self, handle: &str, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        let provider = self.get_provider(handle)?;
        provider.chat(request).await
    }

    /// 发起流式聊天请求
    pub async fn stream_chat(
        &self,
        handle: &str,
        request: ChatRequest,
    ) -> Result<ChatStream, LLMError> {
        let provider = self.get_provider(handle)?;
        provider.stream_chat(request).await
    }

    /// 返回当前已注册的句柄
    pub fn handles(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    fn get_provider(&self, handle: &str) -> Result<DynProvider, LLMError> {
        self.providers
            .get(handle)
            .cloned()
            .ok_or_else(|| LLMError::Validation {
                message: format!("unknown model handle: {handle}"),
            })
    }
}

/// 负责注册 Provider 的 Builder
pub struct LLMClientBuilder {
    providers: HashMap<String, DynProvider>,
}

impl LLMClientBuilder {
    /// 注册一个句柄对应的 Provider
    pub fn register_handle<S: Into<String>>(mut self, handle: S, provider: DynProvider) -> Self {
        self.providers.insert(handle.into(), provider);
        self
    }

    /// 构建最终的 LLMClient
    pub fn build(self) -> LLMClient {
        LLMClient {
            providers: self.providers,
        }
    }
}
