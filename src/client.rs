use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::LLMError;
use crate::provider::{ChatStream, DynProvider};
use crate::types::{CapabilityDescriptor, ChatRequest, ChatResponse};

/// Routes chat requests through the set of registered providers.
///
/// The client stores provider handles in-memory, making it trivial to share one instance
/// across the application and pick the right backend per request.
pub struct LLMClient {
    providers: HashMap<String, DynProvider>,
}

impl LLMClient {
    /// Creates a builder used to register providers before constructing the client.
    ///
    /// The builder enforces unique handles and lets applications inject mocked providers
    /// in tests. Once all handles are registered, call [`LLMClientBuilder::build`] to
    /// obtain an [`LLMClient`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
    /// #         Err(LLMError::NotImplemented { feature: "dummy" })
    /// #     }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
    /// #         Ok(Box::pin(stream::empty()))
    /// #     }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let client = LLMClient::builder()
    ///     .register_handle("dummy", Arc::new(DummyProvider))
    ///     .expect("unique handle")
    ///     .build();
    /// assert_eq!(client.handles(), vec!["dummy".to_string()]);
    /// ```
    pub fn builder() -> LLMClientBuilder {
        LLMClientBuilder {
            providers: HashMap::new(),
        }
    }

    /// Sends a chat request to the provider registered under `handle`.
    ///
    /// The provided [`ChatRequest`] is forwarded verbatim to the selected provider. Use
    /// this method when the full response is required before proceeding.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse, Message, Role};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
    /// #         Err(LLMError::NotImplemented { feature: "dummy" })
    /// #     }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
    /// #         Ok(Box::pin(stream::empty()))
    /// #     }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// # fn client() -> LLMClient {
    /// #     LLMClient::builder()
    /// #         .register_handle("dummy", Arc::new(DummyProvider))
    /// #         .expect("unique handle")
    /// #         .build()
    /// # }
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let request = ChatRequest {
    ///     messages: vec![Message { role: Role::assistant(), name: None, content: Vec::new(), metadata: None }],
    ///     options: Default::default(),
    ///     tools: Vec::new(),
    ///     tool_choice: None,
    ///     response_format: None,
    ///     metadata: None,
    /// };
    /// let result = client().chat("dummy", request).await;
    /// assert!(result.is_err());
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Validation`] when `handle` is not registered, or propagates any
    /// error raised by the underlying provider implementation.
    pub async fn chat(&self, handle: &str, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        let provider = self.get_provider(handle)?;
        provider.chat(request).await
    }

    /// Initiates a streaming chat request.
    ///
    /// This method keeps the HTTP connection open and yields incremental
    /// [`crate::types::ChatChunk`]
    /// values through the returned [`ChatStream`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
    /// #         Err(LLMError::NotImplemented { feature: "dummy" })
    /// #     }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
    /// #         Ok(Box::pin(stream::empty()))
    /// #     }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// # fn client() -> LLMClient {
    /// #     LLMClient::builder()
    /// #         .register_handle("dummy", Arc::new(DummyProvider))
    /// #         .expect("unique handle")
    /// #         .build()
    /// # }
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let request = ChatRequest {
    ///     messages: Vec::new(),
    ///     options: Default::default(),
    ///     tools: Vec::new(),
    ///     tool_choice: None,
    ///     response_format: None,
    ///     metadata: None,
    /// };
    /// let chunks = client()
    ///     .stream_chat("dummy", request)
    ///     .await;
    /// assert!(chunks.is_ok());
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Validation`] when the handle is missing or any transport/provider
    /// level error bubbled up during streaming setup.
    pub async fn stream_chat(
        &self,
        handle: &str,
        request: ChatRequest,
    ) -> Result<ChatStream, LLMError> {
        let provider = self.get_provider(handle)?;
        provider.stream_chat(request).await
    }

    /// Lists every handle currently registered on the client.
    ///
    /// The returned vector is independent from the internal storage, so callers can sort or
    /// mutate it without affecting the client state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let client = LLMClient::builder()
    ///     .register_handle("dummy", Arc::new(DummyProvider))
    ///     .expect("unique handle")
    ///     .build();
    /// assert_eq!(client.handles(), vec!["dummy".to_string()]);
    /// ```
    pub fn handles(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Returns the capability descriptor associated with `handle`.
    ///
    /// Capability descriptors make it easy to filter providers based on streaming support
    /// or multimodal input availability before placing a request.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor {
    /// #         CapabilityDescriptor { supports_stream: true, ..Default::default() }
    /// #     }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let client = LLMClient::builder()
    ///     .register_handle("dummy", Arc::new(DummyProvider))
    ///     .expect("unique handle")
    ///     .build();
    /// let caps = client.capabilities("dummy").expect("registered handle");
    /// assert!(caps.supports_stream);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Validation`] if the handle is unknown.
    pub fn capabilities(&self, handle: &str) -> Result<CapabilityDescriptor, LLMError> {
        let provider = self.get_provider(handle)?;
        Ok(provider.capabilities())
    }

    /// Lists handles whose providers declare tool-calling support.
    ///
    /// Iterate over the result whenever tool execution is required and pick any handle in
    /// the list to guarantee compatibility.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider { tools: bool }
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor {
    /// #         CapabilityDescriptor { supports_tools: self.tools, ..Default::default() }
    /// #     }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let client = LLMClient::builder()
    ///     .register_handle("tools", Arc::new(DummyProvider { tools: true }) as _)
    ///     .expect("tools handle")
    ///     .register_handle("no-tools", Arc::new(DummyProvider { tools: false }) as _)
    ///     .expect("no-tools handle")
    ///     .build();
    /// let mut handles = client.handles_supporting_tools();
    /// handles.sort();
    /// assert_eq!(handles, vec!["tools".to_string()]);
    /// ```
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

    /// Lists handles whose providers support streaming responses.
    ///
    /// Use the resulting handles to ensure [`LLMClient::stream_chat`] succeeds without
    /// falling back to synchronous mode.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider { stream: bool }
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor {
    /// #         CapabilityDescriptor { supports_stream: self.stream, ..Default::default() }
    /// #     }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let client = LLMClient::builder()
    ///     .register_handle("stream", Arc::new(DummyProvider { stream: true }) as _)
    ///     .expect("stream handle")
    ///     .register_handle("sync", Arc::new(DummyProvider { stream: false }) as _)
    ///     .expect("sync handle")
    ///     .build();
    /// let mut handles = client.handles_supporting_stream();
    /// handles.sort();
    /// assert_eq!(handles, vec!["stream".to_string()]);
    /// ```
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

/// Lightweight client abstraction used for dependency injection.
///
/// The trait can be mocked in tests or implemented by alternative clients that still
/// provide the same ergonomic interface for chat and streaming calls.
#[async_trait]
pub trait LLMClientLike: Send + Sync {
    /// Sends a chat request through the trait object.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::{LLMClient, LLMClientLike};
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
    /// #         Err(LLMError::NotImplemented { feature: "dummy" })
    /// #     }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
    /// #         Ok(Box::pin(stream::empty()))
    /// #     }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// # fn build_client() -> LLMClient {
    /// #     LLMClient::builder()
    /// #         .register_handle("dummy", Arc::new(DummyProvider))
    /// #         .expect("unique handle")
    /// #         .build()
    /// # }
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let client = build_client();
    /// let obj: &dyn LLMClientLike = &client;
    /// let request = ChatRequest {
    ///     messages: Vec::new(),
    ///     options: Default::default(),
    ///     tools: Vec::new(),
    ///     tool_choice: None,
    ///     response_format: None,
    ///     metadata: None,
    /// };
    /// let result = obj.chat("dummy", request).await;
    /// assert!(result.is_err());
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates [`LLMError::Validation`] if the handle is missing or any provider failure
    /// bubbled up.
    async fn chat(&self, handle: &str, request: ChatRequest) -> Result<ChatResponse, LLMError>;

    /// Starts a streaming chat request through the trait object.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::{LLMClient, LLMClientLike};
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// # fn build_client() -> LLMClient {
    /// #     LLMClient::builder()
    /// #         .register_handle("dummy", Arc::new(DummyProvider))
    /// #         .expect("unique handle")
    /// #         .build()
    /// # }
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let client = build_client();
    /// let obj: &dyn LLMClientLike = &client;
    /// let result = obj
    ///     .stream_chat("dummy", ChatRequest {
    ///         messages: Vec::new(),
    ///         options: Default::default(),
    ///         tools: Vec::new(),
    ///         tool_choice: None,
    ///         response_format: None,
    ///         metadata: None,
    ///     })
    ///     .await;
    /// assert!(result.is_ok());
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Validation`] when the handle is unknown or forwards any
    /// transport/provider error.
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

/// Builder used to register providers and construct an [`LLMClient`].
pub struct LLMClientBuilder {
    providers: HashMap<String, DynProvider>,
}

impl LLMClientBuilder {
    /// Registers a provider under a unique handle.
    ///
    /// Use this method repeatedly to add each backend. Handles must be unique across the
    /// builder lifetime.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let builder = LLMClient::builder()
    ///     .register_handle("primary", Arc::new(DummyProvider))
    ///     .expect("unique handle");
    /// assert!(builder.build().handles().contains(&"primary".to_string()));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Validation`] if the provided handle already exists.
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

    /// Consumes the builder and returns the configured [`LLMClient`].
    ///
    /// This method finalizes registration, moving the provider map into the client.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::client::LLMClient;
    /// # use kotoba_llm::error::LLMError;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
    /// # use futures_util::stream;
    /// # struct DummyProvider;
    /// # #[async_trait]
    /// # impl LLMProvider for DummyProvider {
    /// #     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> { unreachable!() }
    /// #     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> { Ok(Box::pin(stream::empty())) }
    /// #     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    /// #     fn name(&self) -> &'static str { "dummy" }
    /// # }
    /// let client = LLMClient::builder()
    ///     .register_handle("primary", Arc::new(DummyProvider))
    ///     .expect("unique handle")
    ///     .build();
    /// assert_eq!(client.handles(), vec!["primary".to_string()]);
    /// ```
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

    /// Simple test provider that only exposes capability metadata.
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

    /// Ensures duplicate handles are rejected during registration.
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

    /// Verifies that [`LLMClient`] implements [`LLMClientLike`].
    #[tokio::test]
    async fn llmclient_implements_llmclientlike_trait() {
        let provider = Arc::new(DummyProvider {
            name: "p1",
            caps: CapabilityDescriptor::default(),
        }) as DynProvider;

        let client = LLMClient {
            providers: HashMap::from([("handle".to_string(), provider)]),
        };

        // Invoke chat through the trait object to ensure compilation and behavior.
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

        // DummyProvider always returns a NotImplemented error branch.
        let err = result.expect_err("chat should return error from dummy provider");
        match err {
            LLMError::NotImplemented { feature } => {
                assert_eq!(feature, "dummy_chat");
            }
            other => panic!("unexpected error type from llmclientlike chat: {other:?}"),
        }
    }
}
