use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_core::Stream;
use serde::Serialize;

use crate::error::LLMError;

/// Enumerates HTTP methods understood by the lightweight transport abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

/// Minimal HTTP request representation shared across providers.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
    pub timeout: Option<Duration>,
}

impl HttpRequest {
    /// Builds a POST request with a JSON request body.
    ///
    /// The helper sets the `Content-Type` header to `application/json` and stores the
    /// provided buffer as the body, making it ideal for serialized payloads.
    ///
    /// # Examples
    ///
    /// ```
    /// use kotoba_llm::http::{HttpMethod, HttpRequest};
    ///
    /// let request = HttpRequest::post_json("https://example.com", br"{}".to_vec());
    /// assert_eq!(request.method, HttpMethod::Post);
    /// assert_eq!(request.headers.get("Content-Type"), Some(&"application/json".to_string()));
    /// ```
    pub fn post_json(url: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: HashMap::from([("Content-Type".to_string(), "application/json".to_string())]),
            body: Some(body),
            timeout: None,
        }
    }

    /// Overrides the request headers after construction.
    ///
    /// This is useful when providers need to stamp additional headers or replace
    /// authorization metadata before dispatching the request.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use kotoba_llm::http::HttpRequest;
    ///
    /// let request = HttpRequest::post_json("https://example.com", br"{}".to_vec())
    ///     .with_headers(HashMap::from([("Authorization".into(), "Bearer test".into())]));
    /// assert_eq!(request.headers.get("Authorization"), Some(&"Bearer test".to_string()));
    /// ```
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }
}

/// Minimal HTTP response representation.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Converts the body into a UTF-8 string.
    ///
    /// The method consumes the response and returns the decoded string or a
    /// [`LLMError::Transport`] if the payload contains invalid UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// use kotoba_llm::http::HttpResponse;
    ///
    /// let response = HttpResponse { status: 200, headers: Default::default(), body: b"ok".to_vec() };
    /// assert_eq!(response.into_string().unwrap(), "ok");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LLMError::Transport`] when the body cannot be interpreted as UTF-8.
    pub fn into_string(self) -> Result<String, LLMError> {
        String::from_utf8(self.body).map_err(|err| LLMError::transport(err.to_string()))
    }
}

/// HTTP response that carries a streaming body.
pub struct HttpStreamResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: HttpBodyStream,
}

/// Alias for the body stream returned by [`HttpTransport::send_stream`].
pub type HttpBodyStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, LLMError>> + Send>>;

/// Transport abstraction used to decouple providers from the concrete HTTP client.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    /// Sends a request and resolves when the full response is available.
    ///
    /// # Examples
    ///
    /// ```
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::http::{HttpTransport, HttpRequest, HttpResponse, HttpStreamResponse, HttpBodyStream, HttpMethod};
    /// # use kotoba_llm::error::LLMError;
    /// # use futures_util::stream;
    /// struct MemoryTransport;
    ///
    /// #[async_trait]
    /// impl HttpTransport for MemoryTransport {
    ///     async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError> {
    ///         Ok(HttpResponse { status: 200, headers: request.headers, body: b"ok".to_vec() })
    ///     }
    ///     async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
    ///         Ok(HttpStreamResponse { status: 200, headers: request.headers, body: Box::pin(stream::empty()) })
    ///     }
    /// }
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let transport = MemoryTransport;
    /// let response = transport
    ///     .send(HttpRequest::post_json("https://example.com", br"{}".to_vec()))
    ///     .await
    ///     .unwrap();
    /// assert_eq!(response.status, 200);
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Implementations should map transport failures to [`LLMError::Transport`] and other
    /// issues to the appropriate [`LLMError`] variant.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError>;

    /// Sends a request and returns a streaming body.
    ///
    /// # Examples
    ///
    /// ```
    /// # use async_trait::async_trait;
    /// # use kotoba_llm::http::{HttpTransport, HttpRequest, HttpResponse, HttpStreamResponse, HttpBodyStream};
    /// # use kotoba_llm::error::LLMError;
    /// # use futures_util::{stream, StreamExt};
    /// struct EchoTransport;
    ///
    /// #[async_trait]
    /// impl HttpTransport for EchoTransport {
    ///     async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError> {
    ///         Ok(HttpResponse { status: 200, headers: request.headers, body: request.body.unwrap_or_default() })
    ///     }
    ///     async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
    ///         let stream = stream::once(async move { Ok(request.body.unwrap_or_default()) });
    ///         Ok(HttpStreamResponse { status: 200, headers: request.headers, body: Box::pin(stream) })
    ///     }
    /// }
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let transport = EchoTransport;
    /// let stream = transport
    ///     .send_stream(HttpRequest::post_json("https://example.com", br"{}".to_vec()))
    ///     .await
    ///     .unwrap();
    /// let chunks: Vec<_> = stream.body.collect::<Vec<_>>().await;
    /// assert_eq!(chunks.len(), 1);
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Implementations should return [`LLMError::Transport`] for network failures or
    /// propagate provider-specific errors otherwise.
    async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError>;
}

/// Thread-safe handle to a transport implementation.
pub type DynHttpTransport = Arc<dyn HttpTransport>;

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde::ser;

    /// Transport that panics if `send` or `send_stream` are invoked.
    ///
    /// The helper ensures serialization failures are surfaced before issuing real
    /// network requests.
    struct PanicTransport;

    #[async_trait]
    impl HttpTransport for PanicTransport {
        async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, LLMError> {
            panic!("send should not be called");
        }

        async fn send_stream(&self, _request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
            panic!("send_stream should not be called");
        }
    }

    /// Body type that intentionally fails serialization to trigger validation errors.
    struct NonSerializableBody;

    impl Serialize for NonSerializableBody {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(ser::Error::custom(
                "intentional serialization failure for test",
            ))
        }
    }

    #[tokio::test]
    async fn post_json_with_headers_returns_validation_on_serde_error() {
        let transport = PanicTransport;
        let body = NonSerializableBody;
        let headers = HashMap::new();

        let result = post_json_with_headers(&transport, "http://example.com", headers, &body).await;

        match result {
            Err(LLMError::Validation { message }) => {
                assert!(
                    message.contains("failed to serialize request"),
                    "unexpected validation message: {message}"
                );
            }
            Ok(_) => panic!("expected validation error for non serializable body"),
            other => panic!("unexpected error type: {other:?}"),
        }
    }
}

/// Serializes a body to JSON, attaches headers, and issues a POST request.
///
/// This helper centralizes JSON serialization so each provider can reuse the same logic
/// without duplicating header or error handling.
///
/// # Examples
///
/// ```
/// # use std::collections::HashMap;
/// # use async_trait::async_trait;
/// # use kotoba_llm::http::{post_json_with_headers, HttpTransport, HttpRequest, HttpResponse, HttpStreamResponse};
/// # use kotoba_llm::error::LLMError;
/// # use futures_util::stream;
/// # use serde_json::json;
/// struct MockTransport;
///
/// #[async_trait]
/// impl HttpTransport for MockTransport {
///     async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError> {
///         assert_eq!(request.headers.get("X-Test"), Some(&"ok".to_string()));
///         Ok(HttpResponse { status: 200, headers: request.headers, body: request.body.unwrap_or_default() })
///     }
///     async fn send_stream(&self, _request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
///         panic!("streaming not used in this example");
///     }
/// }
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let mut headers = HashMap::new();
/// headers.insert("X-Test".to_string(), "ok".to_string());
/// let response = post_json_with_headers(
///     &MockTransport,
///     "https://example.com",
///     headers,
///     &json!({"ping": "pong"}),
/// )
/// .await
/// .unwrap();
/// assert_eq!(response.status, 200);
/// # });
/// ```
///
/// # Errors
///
/// Returns [`LLMError::Validation`] if serialization fails or forwards the error raised by
/// [`HttpTransport::send`].
pub async fn post_json_with_headers<T: Serialize>(
    transport: &dyn HttpTransport,
    url: impl Into<String>,
    headers: HashMap<String, String>,
    body: &T,
) -> Result<HttpResponse, LLMError> {
    let payload = serde_json::to_vec(body).map_err(|err| LLMError::Validation {
        message: format!("failed to serialize request: {err}"),
    })?;
    let request = HttpRequest::post_json(url, payload).with_headers(headers);
    transport.send(request).await
}

/// Issues a JSON POST request and returns the streaming response.
///
/// The helper mirrors [`post_json_with_headers`] but calls
/// [`HttpTransport::send_stream`] to support Server-Sent Events and similar protocols.
///
/// # Examples
///
/// ```
/// # use std::collections::HashMap;
/// # use async_trait::async_trait;
/// # use kotoba_llm::http::{post_json_stream_with_headers, HttpTransport, HttpRequest, HttpResponse, HttpStreamResponse, HttpBodyStream};
/// # use kotoba_llm::error::LLMError;
/// # use futures_util::{stream, StreamExt};
/// # use serde_json::json;
/// struct StreamTransport;
///
/// #[async_trait]
/// impl HttpTransport for StreamTransport {
///     async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, LLMError> {
///         panic!("non-streaming call is not used");
///     }
///     async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
///         let first = request.body.unwrap_or_default();
///         let body = stream::once(async move { Ok(first) });
///         Ok(HttpStreamResponse { status: 200, headers: request.headers, body: Box::pin(body) })
///     }
/// }
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let mut headers = HashMap::new();
/// headers.insert("X-Test".to_string(), "ok".to_string());
/// let response = post_json_stream_with_headers(
///     &StreamTransport,
///     "https://example.com",
///     headers,
///     &json!({"hello": "world"}),
/// )
/// .await
/// .unwrap();
/// let collected: Vec<_> = response.body.collect::<Vec<_>>().await;
/// assert_eq!(collected.len(), 1);
/// # });
/// ```
///
/// # Errors
///
/// Returns [`LLMError::Validation`] when serialization fails or propagates any error from
/// [`HttpTransport::send_stream`].
pub async fn post_json_stream_with_headers<T: Serialize>(
    transport: &dyn HttpTransport,
    url: impl Into<String>,
    headers: HashMap<String, String>,
    body: &T,
) -> Result<HttpStreamResponse, LLMError> {
    let payload = serde_json::to_vec(body).map_err(|err| LLMError::Validation {
        message: format!("failed to serialize request: {err}"),
    })?;
    let request = HttpRequest::post_json(url, payload).with_headers(headers);
    transport.send_stream(request).await
}

pub mod reqwest;
