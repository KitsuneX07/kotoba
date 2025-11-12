use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

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

/// 抽象的 HTTP 传输层 便于在测试中注入 Mock
#[async_trait]
pub trait HttpTransport: Send + Sync {
    /// 发送请求并返回响应
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError>;
}

/// 线程安全别名
pub type DynHttpTransport = Arc<dyn HttpTransport>;
