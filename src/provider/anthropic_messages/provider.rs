use std::collections::HashMap;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::LLMError;
use crate::http::{
    DynHttpTransport, HttpResponse, HttpStreamResponse, post_json_stream_with_headers,
    post_json_with_headers,
};
use crate::provider::{ChatStream, LLMProvider};
use crate::types::{CapabilityDescriptor, ChatRequest, ChatResponse};

use super::error::parse_anthropic_error;
use super::request::build_anthropic_body;
use super::response::map_response;
use super::stream::{collect_stream_text, create_stream};
use super::types::AnthropicMessageResponse;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_VERSION: &str = "2023-06-01";

/// Anthropic Messages provider compatible with the Claude 3.x Messages API.
///
/// Converts the unified [`ChatRequest`] format into Anthropic Messages payloads while
/// handling both synchronous and streaming responses.
pub struct AnthropicMessagesProvider {
    pub(crate) transport: DynHttpTransport,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) version: String,
    pub(crate) beta: Option<String>,
    pub(crate) default_model: Option<String>,
}

impl AnthropicMessagesProvider {
    /// Creates a provider with the default base URL and `anthropic-version` header.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::anthropic_messages::AnthropicMessagesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = AnthropicMessagesProvider::new(transport, "test-key");
    /// assert_eq!(provider.name(), "anthropic_messages");
    /// ```
    pub fn new(transport: DynHttpTransport, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            version: DEFAULT_VERSION.to_string(),
            beta: None,
            default_model: None,
        }
    }

    /// Overrides the base URL, which is helpful for proxies or compatibility layers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::anthropic_messages::AnthropicMessagesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = AnthropicMessagesProvider::new(transport, "key")
    ///     .with_base_url("https://anthropic-proxy.local");
    /// assert_eq!(provider.name(), "anthropic_messages");
    /// ```
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Overrides the `anthropic-version` header value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::anthropic_messages::AnthropicMessagesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = AnthropicMessagesProvider::new(transport, "key")
    ///     .with_version("2023-11-01");
    /// assert_eq!(provider.name(), "anthropic_messages");
    /// ```
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Sets the optional `anthropic-beta` header (supports comma-separated beta names).
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::anthropic_messages::AnthropicMessagesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = AnthropicMessagesProvider::new(transport, "key")
    ///     .with_beta("beta1,beta2");
    /// assert_eq!(provider.name(), "anthropic_messages");
    /// ```
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.beta = Some(beta.into());
        self
    }

    /// Configures a default model when requests do not specify one.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::anthropic_messages::AnthropicMessagesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = AnthropicMessagesProvider::new(transport, "key")
    ///     .with_default_model("claude-3-sonnet");
    /// assert!(provider.capabilities().supports_tools);
    /// ```
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    pub(crate) fn endpoint(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/messages")
        } else {
            format!("{base}/v1/messages")
        }
    }

    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), self.api_key.clone());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Accept".to_string(), "application/json".to_string());
        headers.insert("anthropic-version".to_string(), self.version.clone());
        if let Some(beta) = &self.beta {
            headers.insert("anthropic-beta".to_string(), beta.clone());
        }
        headers
    }

    fn resolve_model(&self, request: &ChatRequest) -> Result<String, LLMError> {
        request
            .options
            .model
            .clone()
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| LLMError::Validation {
                message: "model is required for Anthropic Messages".to_string(),
            })
    }

    fn build_request_body(&self, request: &ChatRequest, stream: bool) -> Result<Value, LLMError> {
        let model = self.resolve_model(request)?;
        build_anthropic_body(request, &model, stream)
    }

    async fn send_request(&self, body: Value) -> Result<HttpResponse, LLMError> {
        post_json_with_headers(
            self.transport.as_ref(),
            self.endpoint(),
            self.build_headers(),
            &body,
        )
        .await
    }

    async fn send_stream_request(&self, body: Value) -> Result<HttpStreamResponse, LLMError> {
        post_json_stream_with_headers(
            self.transport.as_ref(),
            self.endpoint(),
            self.build_headers(),
            &body,
        )
        .await
    }

    fn ensure_success(&self, response: HttpResponse) -> Result<String, LLMError> {
        let status = response.status;
        let text = response.into_string()?;
        if (200..300).contains(&status) {
            Ok(text)
        } else {
            Err(parse_anthropic_error(status, &text))
        }
    }

    fn try_parse<T: DeserializeOwned>(&self, text: &str) -> Result<T, LLMError> {
        serde_json::from_str(text).map_err(|err| LLMError::Provider {
            provider: self.name(),
            message: format!("failed to parse Anthropic response: {err}"),
        })
    }
}

#[async_trait]
impl LLMProvider for AnthropicMessagesProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        let body = self.build_request_body(&request, false)?;
        let response = self.send_request(body).await?;
        let text = self.ensure_success(response)?;
        let parsed: AnthropicMessageResponse = self.try_parse(&text)?;
        map_response(parsed, self.name(), self.endpoint())
    }

    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError> {
        let body = self.build_request_body(&request, true)?;
        let response = self.send_stream_request(body).await?;
        if !(200..300).contains(&response.status) {
            let text = collect_stream_text(response.body, self.name()).await?;
            return Err(parse_anthropic_error(response.status, &text));
        }
        Ok(create_stream(response.body, self.name(), self.endpoint()))
    }

    fn capabilities(&self) -> CapabilityDescriptor {
        CapabilityDescriptor {
            supports_stream: true,
            supports_image_input: true,
            supports_audio_input: false,
            supports_video_input: false,
            supports_tools: true,
            // Anthropic Messages has not formally documented JSON mode yet, so leave this conservative.
            supports_structured_output: false,
            supports_parallel_tool_calls: true,
        }
    }

    fn name(&self) -> &'static str {
        "anthropic_messages"
    }
}
