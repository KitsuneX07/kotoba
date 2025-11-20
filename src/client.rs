use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::LLMError;
use crate::provider::{ChatStream, DynProvider};
use crate::types::{CapabilityDescriptor, ChatRequest, ChatResponse};

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

    /// 查询指定句柄的能力描述
    pub fn capabilities(&self, handle: &str) -> Result<CapabilityDescriptor, LLMError> {
        let provider = self.get_provider(handle)?;
        Ok(provider.capabilities())
    }

    /// 返回所有支持工具调用的句柄列表
    pub fn handles_supporting_tools(&self) -> Vec<String> {
        self.providers
            .iter()
            .filter_map(|(handle, provider)| {
                if provider.capabilities().supports_tools {
                    Some(handle.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// 返回所有支持流式输出的句柄列表
    pub fn handles_supporting_stream(&self) -> Vec<String> {
        self.providers
            .iter()
            .filter_map(|(handle, provider)| {
                if provider.capabilities().supports_stream {
                    Some(handle.clone())
                } else {
                    None
                }
            })
            .collect()
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

/// 便于依赖注入与测试的轻量 Client 抽象
#[async_trait]
pub trait LLMClientLike: Send + Sync {
    /// 发送同步聊天请求
    async fn chat(&self, handle: &str, request: ChatRequest) -> Result<ChatResponse, LLMError>;

    /// 发起流式聊天请求
    async fn stream_chat(&self, handle: &str, request: ChatRequest)
    -> Result<ChatStream, LLMError>;
}

#[async_trait]
impl LLMClientLike for LLMClient {
    async fn chat(&self, handle: &str, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        LLMClient::chat(self, handle, request).await
    }

    async fn stream_chat(
        &self,
        handle: &str,
        request: ChatRequest,
    ) -> Result<ChatStream, LLMError> {
        LLMClient::stream_chat(self, handle, request).await
    }
}

/// 负责注册 Provider 的 Builder
pub struct LLMClientBuilder {
    providers: HashMap<String, DynProvider>,
}

impl LLMClientBuilder {
    /// 注册一个句柄对应的 Provider
    pub fn register_handle<S: Into<String>>(
        mut self,
        handle: S,
        provider: DynProvider,
    ) -> Result<Self, LLMError> {
        let handle = handle.into();
        if self.providers.contains_key(&handle) {
            return Err(LLMError::Validation {
                message: format!("duplicate model handle: {handle}"),
            });
        }
        self.providers.insert(handle, provider);
        Ok(self)
    }

    /// 构建最终的 LLMClient
    pub fn build(self) -> LLMClient {
        LLMClient {
            providers: self.providers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::LLMProvider;
    use crate::types::{ChatRequest, ChatResponse};
    use std::sync::Arc;

    /// 简单的测试 Provider 实现 只关注 capabilities
    struct DummyProvider {
        name: &'static str,
        caps: CapabilityDescriptor,
    }

    #[async_trait]
    impl LLMProvider for DummyProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
            Err(LLMError::NotImplemented {
                feature: "dummy_chat",
            })
        }

        async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
            Err(LLMError::NotImplemented {
                feature: "dummy_stream",
            })
        }

        fn capabilities(&self) -> CapabilityDescriptor {
            self.caps.clone()
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    #[test]
    fn capabilities_returns_descriptor_for_handle() {
        let provider = DummyProvider {
            name: "dummy",
            caps: CapabilityDescriptor {
                supports_stream: true,
                supports_image_input: false,
                supports_audio_input: false,
                supports_video_input: false,
                supports_tools: true,
                supports_structured_output: false,
                supports_parallel_tool_calls: false,
            },
        };

        let client = LLMClient {
            providers: HashMap::from([("p1".to_string(), Arc::new(provider) as DynProvider)]),
        };

        let caps = client.capabilities("p1").expect("capabilities");
        assert!(caps.supports_stream);
        assert!(caps.supports_tools);
        assert!(!caps.supports_image_input);
    }

    #[test]
    fn capabilities_returns_error_for_unknown_handle() {
        let client = LLMClient {
            providers: HashMap::new(),
        };

        let err = client.capabilities("missing").expect_err("should fail");
        match err {
            LLMError::Validation { message } => {
                assert!(
                    message.contains("missing"),
                    "unexpected validation message: {message}"
                );
            }
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn handles_supporting_tools_filters_by_capability() {
        let providers: HashMap<String, DynProvider> = HashMap::from([
            (
                "tools-and-stream".to_string(),
                Arc::new(DummyProvider {
                    name: "p1",
                    caps: CapabilityDescriptor {
                        supports_stream: true,
                        supports_image_input: false,
                        supports_audio_input: false,
                        supports_video_input: false,
                        supports_tools: true,
                        supports_structured_output: false,
                        supports_parallel_tool_calls: false,
                    },
                }) as DynProvider,
            ),
            (
                "stream-only".to_string(),
                Arc::new(DummyProvider {
                    name: "p2",
                    caps: CapabilityDescriptor {
                        supports_stream: true,
                        supports_image_input: false,
                        supports_audio_input: false,
                        supports_video_input: false,
                        supports_tools: false,
                        supports_structured_output: false,
                        supports_parallel_tool_calls: false,
                    },
                }) as DynProvider,
            ),
        ]);

        let client = LLMClient { providers };
        let mut handles = client.handles_supporting_tools();
        handles.sort();

        assert_eq!(handles, vec!["tools-and-stream".to_string()]);
    }

    #[test]
    fn handles_supporting_stream_filters_by_capability() {
        let providers: HashMap<String, DynProvider> = HashMap::from([
            (
                "stream-enabled".to_string(),
                Arc::new(DummyProvider {
                    name: "p1",
                    caps: CapabilityDescriptor {
                        supports_stream: true,
                        supports_image_input: false,
                        supports_audio_input: false,
                        supports_video_input: false,
                        supports_tools: false,
                        supports_structured_output: false,
                        supports_parallel_tool_calls: false,
                    },
                }) as DynProvider,
            ),
            (
                "no-stream".to_string(),
                Arc::new(DummyProvider {
                    name: "p2",
                    caps: CapabilityDescriptor {
                        supports_stream: false,
                        supports_image_input: false,
                        supports_audio_input: false,
                        supports_video_input: false,
                        supports_tools: true,
                        supports_structured_output: false,
                        supports_parallel_tool_calls: false,
                    },
                }) as DynProvider,
            ),
        ]);

        let client = LLMClient { providers };
        let mut handles = client.handles_supporting_stream();
        handles.sort();

        assert_eq!(handles, vec!["stream-enabled".to_string()]);
    }

    /// 重复句柄应当在注册阶段被显式拒绝
    #[test]
    fn register_handle_rejects_duplicate_handle() {
        let provider1 = Arc::new(DummyProvider {
            name: "p1",
            caps: CapabilityDescriptor::default(),
        }) as DynProvider;
        let provider2 = Arc::new(DummyProvider {
            name: "p2",
            caps: CapabilityDescriptor::default(),
        }) as DynProvider;

        let builder = LLMClient::builder();
        let builder = builder
            .register_handle("duplicate", provider1)
            .expect("first registration should succeed");

        let result = builder.register_handle("duplicate", provider2);
        let err = match result {
            Ok(_) => panic!("expected duplicate handle error"),
            Err(err) => err,
        };

        match err {
            LLMError::Validation { message } => {
                assert!(
                    message.contains("duplicate model handle: duplicate"),
                    "unexpected validation message for duplicate handle: {message}"
                );
            }
            other => panic!("unexpected error type for duplicate handle: {other:?}"),
        }
    }

    /// 通过 LLMClientLike trait 对 LLMClient 进行调用
    #[tokio::test]
    async fn llmclient_implements_llmclientlike_trait() {
        let provider = Arc::new(DummyProvider {
            name: "p1",
            caps: CapabilityDescriptor::default(),
        }) as DynProvider;

        let client = LLMClient {
            providers: HashMap::from([("handle".to_string(), provider)]),
        };

        // 通过 trait 对象调用 chat 确认编译与运行行为正常
        let trait_obj: &dyn LLMClientLike = &client;
        let result = trait_obj
            .chat(
                "handle",
                ChatRequest {
                    messages: Vec::new(),
                    options: Default::default(),
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    metadata: None,
                },
            )
            .await;

        // DummyProvider 总是返回 NotImplemented
        let err = result.expect_err("chat should return error from dummy provider");
        match err {
            LLMError::NotImplemented { feature } => {
                assert_eq!(feature, "dummy_chat");
            }
            other => panic!("unexpected error type from llmclientlike chat: {other:?}"),
        }
    }
}
