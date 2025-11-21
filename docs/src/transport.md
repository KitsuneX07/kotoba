# HTTP 传输与测试

## 抽象层

`src/http/mod.rs` 定义的 `HttpTransport` trait 只有两个异步方法：

```rust
#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError>;
    async fn send_stream(&self, request: HttpRequest) -> Result<HttpStreamResponse, LLMError>;
}
```

- `HttpRequest` 记录 method、url、headers、body、timeout；
- `HttpResponse`/`HttpStreamResponse` 提供 `into_string()`、`HttpBodyStream` 等便捷方法；
- 工具函数 `post_json_with_headers`、`post_json_stream_with_headers` 帮助 Provider 构造 `POST + JSON` 请求。

## 默认实现

`ReqwestTransport`（`src/http/reqwest.rs`）封装 `reqwest::Client`，负责：

- 根据 `HttpMethod` 构建请求并拷贝 header/body；
- 在 `send_stream` 中把 `bytes_stream` 转换成 `HttpBodyStream`，并把 `reqwest::Error` 统一映射为 `LLMError::Transport`；
- `default_dyn_transport()` 返回 `Arc<ReqwestTransport>`，日常使用时直接传给 Provider 即可。

## Mock 与测试

由于 Provider 只依赖 `DynHttpTransport`，可以在测试中注入伪实现：

```rust
use async_trait::async_trait;
use futures_util::stream;
use kotoba::http::{HttpTransport, HttpRequest, HttpResponse, HttpStreamResponse, HttpBodyStream};
use kotoba::error::LLMError;

struct MockTransport;

#[async_trait]
impl HttpTransport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, LLMError> {
        assert!(request.url.contains("chat/completions"));
        Ok(HttpResponse {
            status: 200,
            headers: Default::default(),
            body: br#"{\"id\":\"mock\"}"#.to_vec(),
        })
    }

    async fn send_stream(&self, _request: HttpRequest) -> Result<HttpStreamResponse, LLMError> {
        let stream = stream::once(async { Ok(br"data: [DONE]\n\n".to_vec()) });
        Ok(HttpStreamResponse {
            status: 200,
            headers: Default::default(),
            body: Box::pin(stream) as HttpBodyStream,
        })
    }
}
```

借助 mock 可以：

- 验证请求 JSON 是否包含期望字段（通过深拷贝 `HttpRequest.body`）；
- 为 `stream_chat` 提供自定义 SSE 序列，确保流式解析逻辑的单元测试不依赖公网；
- 在回归测试里注入故障（如返回 429 或畸形 JSON）以覆盖错误分支。

## 超时与重试

`HttpRequest` 自带 `timeout: Option<Duration>` 字段，Provider 可以在构造请求时填入（目前默认留空，由上层 HTTP 客户端控制）。如果需要跨 Provider 统一设置，推荐：

1. 自定义 `reqwest::Client`，在 builder 中配置超时与代理，然后传入 `ReqwestTransport::new(client)`；
2. 或者实现一个包装器，在调用 `inner.send()` 前设置 `request.timeout`、注入 trace header、记录 metrics。

这样既不会污染 Provider 的业务逻辑，也让网络策略集中可控。
