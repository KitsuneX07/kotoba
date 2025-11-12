use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, Method};

use crate::error::LLMError;

use super::{
    DynHttpTransport, HttpBodyStream, HttpMethod, HttpRequest, HttpResponse, HttpStreamResponse,
    HttpTransport,
};

/// 基于 reqwest 的默认 HttpTransport
pub struct ReqwestTransport {
    client: Client,
}

impl ReqwestTransport {
    /// 使用自定义 reqwest::Client
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// 创建默认配置
    pub fn default_client() -> Result<Self, LLMError> {
        Client::builder()
            .build()
            .map(Self::new)
            .map_err(|err| LLMError::transport(format!("failed to create reqwest client: {err}")))
    }

    fn method(method: HttpMethod) -> Method {
        match method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
            HttpMethod::Put => Method::PUT,
            HttpMethod::Patch => Method::PATCH,
            HttpMethod::Delete => Method::DELETE,
        }
    }

    fn build_request(&self, mut request: HttpRequest) -> Result<reqwest::RequestBuilder, LLMError> {
        let method = Self::method(request.method);
        let mut builder = self.client.request(method, &request.url);

        if let Some(timeout) = request.timeout {
            builder = builder.timeout(timeout);
        }

        for (name, value) in request.headers.drain() {
            let header_name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|err| LLMError::transport(format!("invalid header name: {err}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value).map_err(|err| {
                LLMError::transport(format!("invalid header value for {header_name}: {err}"))
            })?;
            builder = builder.header(header_name, header_value);
        }

        if let Some(body) = request.body.take() {
            builder = builder.body(body);
        }

        Ok(builder)
    }

    fn headers_to_map(headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
        headers
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_string(),
                    value.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        ReqwestTransport::default_client().expect("failed to initialize default reqwest transport")
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError> {
        let response = self
            .build_request(request)?
            .send()
            .await
            .map_err(|err| LLMError::transport(err.to_string()))?;

        let status = response.status().as_u16();
        let headers = Self::headers_to_map(response.headers());
        let body = response
            .bytes()
            .await
            .map_err(|err| LLMError::transport(err.to_string()))?
            .to_vec();

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
        let response = self
            .build_request(request)?
            .send()
            .await
            .map_err(|err| LLMError::transport(err.to_string()))?;

        let status = response.status().as_u16();
        let headers = Self::headers_to_map(response.headers());
        let stream = response.bytes_stream().map(|chunk| {
            chunk
                .map(|bytes| bytes.to_vec())
                .map_err(|err| LLMError::transport(err.to_string()))
        });
        let body: HttpBodyStream = Box::pin(stream);

        Ok(HttpStreamResponse {
            status,
            headers,
            body,
        })
    }
}

/// 便捷构造线程安全 Transport
pub fn default_dyn_transport() -> Result<DynHttpTransport, LLMError> {
    Ok(Arc::new(ReqwestTransport::default()))
}
