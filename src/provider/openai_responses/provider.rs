use std::collections::HashMap;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::config::RequestPatch;
use crate::error::LLMError;
use crate::http::{
    DynHttpTransport, HttpResponse, HttpStreamResponse, post_json_stream_with_headers,
    post_json_with_headers,
};
use crate::provider::{ChatStream, LLMProvider, retry::retry_after_from_headers};
use crate::types::{CapabilityDescriptor, ChatRequest, ChatResponse};

use super::error::parse_openai_responses_error;
use super::request::build_openai_responses_body;
use super::response::map_responses_response;
use super::stream::{collect_stream_text, create_stream};
use super::types::OpenAiResponsesResponse;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// OpenAI Responses provider implementation.
///
/// Translates [`ChatRequest`] structures into OpenAI Responses payloads and handles both
/// synchronous and streaming flows, including tool invocation support.
pub struct OpenAiResponsesProvider {
    pub(crate) transport: DynHttpTransport,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) organization: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) default_model: Option<String>,
    pub(crate) request_patch: Option<RequestPatch>,
}

impl OpenAiResponsesProvider {
    /// Creates a provider targeting the default `https://api.openai.com` endpoint.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::openai_responses::OpenAiResponsesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = OpenAiResponsesProvider::new(transport, "key");
    /// assert_eq!(provider.name(), "openai_responses");
    /// ```
    pub fn new(transport: DynHttpTransport, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            organization: None,
            project: None,
            default_model: None,
            request_patch: None,
        }
    }

    /// Overrides the base URL, useful for proxies or gateways.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::openai_responses::OpenAiResponsesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = OpenAiResponsesProvider::new(transport, "key")
    ///     .with_base_url("https://proxy.local");
    /// assert_eq!(provider.name(), "openai_responses");
    /// ```
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the optional `OpenAI-Organization` header.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::openai_responses::OpenAiResponsesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = OpenAiResponsesProvider::new(transport, "key").with_organization("org_123");
    /// assert_eq!(provider.name(), "openai_responses");
    /// ```
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Sets the optional `OpenAI-Project` header.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::openai_responses::OpenAiResponsesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = OpenAiResponsesProvider::new(transport, "key").with_project("proj_alpha");
    /// assert_eq!(provider.name(), "openai_responses");
    /// ```
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Configures a default model when requests omit one.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kotoba_llm::provider::openai_responses::OpenAiResponsesProvider;
    /// # use kotoba_llm::provider::LLMProvider;
    /// # use kotoba_llm::http::reqwest::default_dyn_transport;
    /// let transport = default_dyn_transport().expect("transport");
    /// let provider = OpenAiResponsesProvider::new(transport, "key")
    ///     .with_default_model("gpt-4.1-mini");
    /// assert_eq!(provider.name(), "openai_responses");
    /// ```
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Constructs a provider from a [`ModelConfig`].
    ///
    /// This method is used by the macro-driven provider registration system to build
    /// providers from declarative configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The model configuration containing credentials and settings
    /// * `transport` - The HTTP transport implementation to use
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Auth`] when credentials are missing or invalid.
    pub fn from_model_config(
        config: &crate::config::ModelConfig,
        transport: DynHttpTransport,
    ) -> Result<Self, LLMError> {
        use crate::config::Credential;

        let api_key = match &config.credential {
            Credential::ApiKey { key, .. } => key.clone(),
            Credential::Bearer { token } => token.clone(),
            Credential::ServiceAccount { .. } => {
                return Err(LLMError::Auth {
                    message:
                        "provider openai_responses does not support service account credential"
                            .to_string(),
                });
            }
            Credential::None => {
                return Err(LLMError::Auth {
                    message: "provider openai_responses requires credential".to_string(),
                });
            }
        };

        let mut provider = Self::new(transport, api_key);

        if let Some(base_url) = &config.base_url {
            provider = provider.with_base_url(base_url.clone());
        }

        if let Some(model) = &config.default_model {
            provider = provider.with_default_model(model.clone());
        }

        if let Some(Value::String(org)) = config.extra.get("organization") {
            provider = provider.with_organization(org.clone());
        }

        if let Some(Value::String(project)) = config.extra.get("project") {
            provider = provider.with_project(project.clone());
        }

        provider.request_patch = config.patch.clone();

        Ok(provider)
    }

    pub(crate) fn endpoint(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}/responses")
        } else {
            format!("{base}/v1/responses")
        }
    }

    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        );
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Accept".to_string(), "application/json".to_string());
        if let Some(org) = &self.organization {
            headers.insert("OpenAI-Organization".to_string(), org.clone());
        }
        if let Some(project) = &self.project {
            headers.insert("OpenAI-Project".to_string(), project.clone());
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
                message: "model is required for OpenAI Responses".to_string(),
            })
    }

    fn build_request_body(&self, request: &ChatRequest, stream: bool) -> Result<Value, LLMError> {
        let model = self.resolve_model(request)?;
        build_openai_responses_body(request, &model, stream)
    }

    async fn send_request(&self, body: Value) -> Result<HttpResponse, LLMError> {
        let mut url = self.endpoint();
        let mut headers = self.build_headers();
        let mut body = body;
        self.apply_patch(&mut url, &mut headers, &mut body);
        post_json_with_headers(self.transport.as_ref(), url, headers, &body).await
    }

    async fn send_stream_request(&self, body: Value) -> Result<HttpStreamResponse, LLMError> {
        let mut url = self.endpoint();
        let mut headers = self.build_headers();
        let mut body = body;
        self.apply_patch(&mut url, &mut headers, &mut body);
        post_json_stream_with_headers(self.transport.as_ref(), url, headers, &body).await
    }

    fn ensure_success(&self, response: HttpResponse) -> Result<String, LLMError> {
        let HttpResponse {
            status,
            headers,
            body,
        } = response;
        let text = String::from_utf8(body).map_err(|err| LLMError::transport(err.to_string()))?;
        if (200..300).contains(&status) {
            Ok(text)
        } else {
            Err(parse_openai_responses_error(
                status,
                &text,
                retry_after_from_headers(&headers),
            ))
        }
    }

    fn try_parse<T: DeserializeOwned>(&self, text: &str) -> Result<T, LLMError> {
        serde_json::from_str(text).map_err(|err| LLMError::Provider {
            provider: self.name(),
            message: format!("failed to parse OpenAI Responses response: {err}"),
        })
    }

    fn apply_patch(
        &self,
        url: &mut String,
        headers: &mut HashMap<String, String>,
        body: &mut Value,
    ) {
        if let Some(patch) = &self.request_patch {
            patch.apply(url, headers, body);
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAiResponsesProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        let body = self.build_request_body(&request, false)?;
        let response = self.send_request(body).await?;
        let text = self.ensure_success(response)?;
        let parsed: OpenAiResponsesResponse = self.try_parse(&text)?;
        map_responses_response(parsed, self.name(), self.endpoint())
    }

    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError> {
        let body = self.build_request_body(&request, true)?;
        let response = self.send_stream_request(body).await?;
        let HttpStreamResponse {
            status,
            headers,
            body,
        } = response;
        if !(200..300).contains(&status) {
            let text = collect_stream_text(body, self.name()).await?;
            return Err(parse_openai_responses_error(
                status,
                &text,
                retry_after_from_headers(&headers),
            ));
        }
        Ok(create_stream(body, self.name(), self.endpoint()))
    }

    fn capabilities(&self) -> CapabilityDescriptor {
        CapabilityDescriptor {
            supports_stream: true,
            supports_image_input: true,
            // Audio and video input coverage is not fully documented in Responses yet,
            // so these flags stay conservative for now.
            supports_audio_input: false,
            supports_video_input: false,
            supports_tools: true,
            supports_structured_output: true,
            supports_parallel_tool_calls: true,
        }
    }

    fn name(&self) -> &'static str {
        "openai_responses"
    }
}
