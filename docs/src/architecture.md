# 核心类型与架构

## 模块速览

| 模块 | 作用 |
| --- | --- |
| `src/types` | 定义 `Role`、`Message`、`ContentPart` 到 `ChatRequest`、`ChatResponse`、`ChatChunk` 的全量数据结构，覆盖文本、多模态、工具、推理与流式事件。 |
| `src/provider` | 暴露 `LLMProvider` trait 与具体供应商实现，负责将统一模型映射为厂商 API 请求并解析响应。 |
| `src/client` | 提供 `LLMClient` 与 `LLMClientBuilder`，路由 handle → Provider，支持能力查询及工具/流式筛选。 |
| `src/config` | 用 `ModelConfig`/`ProviderKind`/`Credential` 表示外部配置，并提供 `build_client_from_configs` 批量注册 Provider。 |
| `src/http` | 定义轻量 `HttpTransport` 抽象与 `ReqwestTransport` 默认实现，便于切换或注入 mock。 |
| `src/error` | 聚合所有错误为 `LLMError`，并提供 `transport()`、`provider()` 等便捷构造。 |

## 多模态与工具建模

`Message` 通过 `Vec<ContentPart>` 支持以下类型：

- 文本：`ContentPart::Text(TextContent)`；
- 图片：`ImageContent` 支持 URL、Base64、文件 ID，并带 `ImageDetail` 枚举；
- 音频 / 视频：`AudioContent`、`VideoContent` 结合 `MediaSource::Inline/FileId/Url`；
- 文件引用、原始 JSON 数据；
- 工具调用 (`ToolCall`) 与工具结果 (`ToolResult`)，二者在不同 Provider 中被各自的 request mapper 处理。

`ChatRequest` 附带 `ChatOptions`（温度、`top_p`、`max_output_tokens`、penalty、`parallel_tool_calls`、`ReasoningOptions`、`extra`），并行工具策略由 `ToolChoice` 决定，输出格式通过 `ResponseFormat` 声明（文本 / JSON / JSON Schema / 自定义）。

`ChatResponse` 统一封装 `OutputItem`（消息、工具、工具结果、推理文本、自定义 payload）、`TokenUsage`、`FinishReason` 及 `ProviderMetadata`。流式场景使用 `ChatChunk` + `ChatEvent` 描述增量文本/工具 delta，保持与同步响应相同的语义。

## Provider 抽象

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError>;
    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError>;
    fn capabilities(&self) -> CapabilityDescriptor;
    fn name(&self) -> &'static str;
}
```

每个 Provider 模块拆成 `provider.rs`（实现 trait）、`request.rs`（构建 JSON）、`response.rs`（解析响应为统一类型）、`stream.rs`（SSE/Chunk 解析）、`error.rs`（HTTP 错误解析）、`types.rs`（中间结构）。这种分层让新增 Provider 只需补齐映射逻辑即可。

`CapabilityDescriptor` 用于声明 `supports_stream`、`supports_image_input` 等能力，`LLMClient` 通过它进行 handle 过滤。

## 客户端路由

`LLMClient` 内部维护 `HashMap<String, DynProvider>`：

- `chat(handle, request)` 与 `stream_chat(handle, request)` 只负责定位 Provider 并转发；
- `handles()` 返回 handle 列表；
- `capabilities(handle)` 获取特性描述；
- `handles_supporting_tools()` / `handles_supporting_stream()` 根据 `CapabilityDescriptor` 自动筛选。

`LLMClientBuilder` 提供 `register_handle(handle, Arc<dyn LLMProvider>)`，在 `build()` 时做重复 handle 校验并返回可用客户端。测试中可以注入简单的 `LLMProvider` stub 验证路由逻辑。

## 配置与凭证

`ModelConfig` 暴露以下字段：`handle`、`provider`、`credential`、`default_model`、`base_url`、`extra`。`ProviderKind` 枚举包含四个当前实现。`Credential` 支持 `ApiKey`（可自定义 header）、`Bearer`、`ServiceAccount`（仅做校验，当前 Provider 均不接受）、`None`。`build_client_from_configs` 按序构造 Provider 并注册 handle，遇到缺少凭证、重复 handle、或 Service Account 等不被支持的 credential 时抛出 `LLMError::Auth/Validation`。

## HTTP 抽象

`HttpTransport` 提供 `send(HttpRequest) -> HttpResponse` 与 `send_stream(HttpRequest) -> HttpStreamResponse`。默认的 `ReqwestTransport` 实现在 `http/reqwest.rs`，但调用方可注入自定义实现以满足：

- 统一代理/超时/重试策略；
- 在测试中返回固定 JSON 或模拟 SSE。

`post_json_with_headers`、`post_json_stream_with_headers` 小工具隐藏了重复的 header 注入逻辑，使 Provider 实现专注于业务映射。

## 错误处理

所有错误通过 `LLMError` 传播：

- 网络层：`Transport`；
- 鉴权：`Auth`；
- 速率限制：`RateLimit`（附带 `retry_after`）；
- 请求校验：`Validation`；
- 能力不支持：`UnsupportedFeature`；
- 厂商返回：`Provider`；
- 预留占位：`NotImplemented`、`Unknown`。

在 Provider 内可以直接使用 `LLMError::transport()` 和 `LLMError::provider()` 保持信息格式一致，便于上层日志与重试策略集中处理。
