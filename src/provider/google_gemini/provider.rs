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

use super::error::parse_gemini_error;
use super::request::build_gemini_body;
use super::response::map_response;
use super::stream::{collect_stream_text, create_stream};
use super::types::GeminiGenerateContentResponse;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Google Gemini GenerateContent provider implementation.
pub struct GoogleGeminiProvider {
    pub(crate) transport: DynHttpTransport,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) default_model: Option<String>,
    pub(crate) request_patch: Option<RequestPatch>,
}

impl GoogleGeminiProvider {
    /// Creates a provider that targets the default Google Generative Language endpoint.
    pub fn new(transport: DynHttpTransport, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            default_model: None,
            request_patch: None,
        }
    }

    /// Overrides the base URL, making it easier to point at proxies or compatibility layers.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets a default model such as `gemini-2.0-flash` when the request omits one.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Constructs a provider from a [`crate::config::ModelConfig`].
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
                    message: "provider google_gemini does not support service account credential"
                        .to_string(),
                });
            }
            Credential::None => {
                return Err(LLMError::Auth {
                    message: "provider google_gemini requires credential".to_string(),
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

        provider.request_patch = config.patch.clone();

        Ok(provider)
    }

    /// Builds the non-streaming endpoint URL for GenerateContent.
    pub(crate) fn endpoint(&self, model: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let model_path = normalize_model(model);
        if base.ends_with("/v1beta") {
            format!("{base}/{model_path}:generateContent")
        } else {
            format!("{base}/v1beta/{model_path}:generateContent")
        }
    }

    /// Builds the streaming endpoint URL (SSE) for GenerateContent.
    pub(crate) fn stream_endpoint(&self, model: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let model_path = normalize_model(model);
        if base.ends_with("/v1beta") {
            format!("{base}/{model_path}:streamGenerateContent?alt=sse")
        } else {
            format!("{base}/v1beta/{model_path}:streamGenerateContent?alt=sse")
        }
    }

    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("x-goog-api-key".to_string(), self.api_key.clone());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Accept".to_string(), "application/json".to_string());
        headers
    }

    fn resolve_model(&self, request: &ChatRequest) -> Result<String, LLMError> {
        request
            .options
            .model
            .clone()
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| LLMError::Validation {
                message: "model is required for Google Gemini".to_string(),
            })
    }

    fn build_request_body(
        &self,
        request: &ChatRequest,
        model: &str,
        stream: bool,
    ) -> Result<Value, LLMError> {
        build_gemini_body(request, model, stream)
    }

    async fn send_request(&self, url: String, body: Value) -> Result<HttpResponse, LLMError> {
        let mut url = url;
        let mut headers = self.build_headers();
        let mut body = body;
        self.apply_patch(&mut url, &mut headers, &mut body);
        post_json_with_headers(self.transport.as_ref(), url, headers, &body).await
    }

    async fn send_stream_request(
        &self,
        url: String,
        body: Value,
    ) -> Result<HttpStreamResponse, LLMError> {
        let mut url = url;
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
            Err(parse_gemini_error(
                status,
                &text,
                retry_after_from_headers(&headers),
            ))
        }
    }

    fn try_parse<T: DeserializeOwned>(&self, text: &str) -> Result<T, LLMError> {
        serde_json::from_str(text).map_err(|err| LLMError::Provider {
            provider: self.name(),
            message: format!("failed to parse Gemini response: {err}"),
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

fn normalize_model(model: &str) -> String {
    if model.starts_with("models/") {
        model.to_string()
    } else {
        format!("models/{model}")
    }
}

#[async_trait]
impl LLMProvider for GoogleGeminiProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError> {
        let model = self.resolve_model(&request)?;
        let endpoint = self.endpoint(&model);
        let body = self.build_request_body(&request, &model, false)?;
        let response = self.send_request(endpoint.clone(), body).await?;
        let text = self.ensure_success(response)?;
        let parsed: GeminiGenerateContentResponse = self.try_parse(&text)?;
        map_response(parsed, self.name(), endpoint)
    }

    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError> {
        let model = self.resolve_model(&request)?;
        let endpoint = self.stream_endpoint(&model);
        let body = self.build_request_body(&request, &model, true)?;
        let response = self.send_stream_request(endpoint.clone(), body).await?;
        let HttpStreamResponse {
            status,
            headers,
            body,
        } = response;
        if !(200..300).contains(&status) {
            let text = collect_stream_text(body, self.name()).await?;
            return Err(parse_gemini_error(
                status,
                &text,
                retry_after_from_headers(&headers),
            ));
        }
        Ok(create_stream(body, self.name(), endpoint))
    }

    fn capabilities(&self) -> CapabilityDescriptor {
        CapabilityDescriptor {
            supports_stream: true,
            supports_image_input: true,
            supports_audio_input: true,
            supports_video_input: true,
            supports_tools: true,
            supports_structured_output: true,
            supports_parallel_tool_calls: true,
        }
    }

    fn name(&self) -> &'static str {
        "google_gemini"
    }
}
