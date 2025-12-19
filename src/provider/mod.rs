use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;

use crate::error::LLMError;
use crate::types::{CapabilityDescriptor, ChatChunk, ChatRequest, ChatResponse};

pub mod anthropic_messages;
pub mod google_gemini;
pub mod macros;
pub mod openai_chat;
pub mod openai_responses;
pub(crate) mod retry;

pub use retry::{RetryConfig, RetryableLLMProvider};

/// Stream alias returned by provider implementations for incremental responses.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatChunk, LLMError>> + Send>>;

/// Trait implemented by every provider integration.
///
/// Providers translate the unified [`ChatRequest`] into a vendor-specific HTTP call and
/// map the response back to [`ChatResponse`] or [`ChatChunk`] events.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Submits a request and waits for the full response body.
    ///
    /// # Examples
    ///
    /// ```
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{ChatRequest, ChatResponse, CapabilityDescriptor};
    /// # use kotoba_llm::error::LLMError;
    /// # use futures_util::stream;
    /// struct RejectingProvider;
    ///
    /// #[async_trait]
    /// impl LLMProvider for RejectingProvider {
    ///     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
    ///         Err(LLMError::NotImplemented { feature: "chat" })
    ///     }
    ///     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
    ///         Ok(Box::pin(stream::empty()))
    ///     }
    ///     fn capabilities(&self) -> CapabilityDescriptor { CapabilityDescriptor::default() }
    ///     fn name(&self) -> &'static str { "rejecting" }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Implementations should propagate any [`LLMError`] raised while invoking the remote
    /// provider.
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError>;

    /// Submits a request and returns a stream of incremental events.
    ///
    /// # Examples
    ///
    /// ```
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::provider::{LLMProvider, ChatStream};
    /// # use kotoba_llm::types::{ChatRequest, ChatResponse, CapabilityDescriptor};
    /// # use kotoba_llm::error::LLMError;
    /// # use futures_util::stream;
    /// struct StreamingProvider;
    ///
    /// #[async_trait]
    /// impl LLMProvider for StreamingProvider {
    ///     async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
    ///         Err(LLMError::NotImplemented { feature: "chat" })
    ///     }
    ///     async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
    ///         Ok(Box::pin(stream::empty()))
    ///     }
    ///     fn capabilities(&self) -> CapabilityDescriptor {
    ///         CapabilityDescriptor { supports_stream: true, ..Default::default() }
    ///     }
    ///     fn name(&self) -> &'static str { "streaming" }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Implementations should surface transport issues via [`LLMError::Transport`] and
    /// vendor-specific errors through [`LLMError::Provider`].
    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError>;

    /// Returns the provider's capability descriptor.
    fn capabilities(&self) -> CapabilityDescriptor;

    /// Returns the provider identifier used in logs and error reporting.
    fn name(&self) -> &'static str;
}

/// Thread-safe handle to a provider implementation.
pub type DynProvider = Arc<dyn LLMProvider>;
