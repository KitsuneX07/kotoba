use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_core::Stream;
use serde::Serialize;

use crate::error::LLMError;

/// HTTP 方法枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

/// HTTP 请求
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
    pub timeout: Option<Duration>,
}

impl HttpRequest {
    /// 构建 JSON POST 请求
    pub fn post_json(url: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: HashMap::from([("Content-Type".to_string(), "application/json".to_string())]),
            body: Some(body),
            timeout: None,
        }
    }

    /// 替换请求头 便于在构造后统一设置 Provider 定制 header
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }
}

/// HTTP 响应
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// 将响应体转换为 UTF-8 字符串
    pub fn into_string(self) -> Result<String, LLMError> {
        String::from_utf8(self.body).map_err(|err| LLMError::transport(err.to_string()))
    }
}

/// HTTP 流式响应
pub struct HttpStreamResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: HttpBodyStream,
}

/// 流式响应体
pub type HttpBodyStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, LLMError>> + Send>>;

/// 抽象的 HTTP 传输层 便于在测试中注入 Mock
#[async_trait]
pub trait HttpTransport: Send + Sync {
    /// 发送请求并返回响应
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError>;

    /// 以流式方式发送请求并持续接收响应体
    async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError>;
}

/// 线程安全别名
pub type DynHttpTransport = Arc<dyn HttpTransport>;

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde::ser;

    /// 一个在 send/send_stream 被调用时直接 panic 的 Transport
    /// 用于验证序列化失败时不会触发底层请求发送
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

    /// 自定义一个总是序列化失败的类型 用于触发序列化错误分支
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

/// 使用统一逻辑发送 JSON POST 请求 方便 Provider 复用
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

/// 使用统一逻辑发送 JSON POST 流式请求
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
