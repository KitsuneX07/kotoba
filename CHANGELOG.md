# 更新日志

## 0.2.0 - 2025-12-19

- `register_providers!` 宏自动生成 `ProviderKind`、`build_provider_from_config` 与 `list_provider_types`（`src/provider/macros.rs`），`config::build_client_from_configs` 因此能够一次性实例化 OpenAI/Anthropic/Gemini 等全部 Provider，并对重复 handle、缺失凭证或不支持的 Service Account 做出显式校验（`src/config.rs`）
- `ModelConfig.patch` 允许覆盖 URL、深度合并 Body、增删 Header 及删除不兼容字段，可用于企业代理、灰度参数或第三方 OpenAI 兼容层，无需改动业务代码（`src/config.rs`、`docs/src/client-config.md`）
- `LLMClient` 新增 `chat_with_retry`（结合 `RetryConfig` 与 `RetryableLLMProvider`）针对 `RateLimit`/`Transport` 自动指数退避，同时提供 `capabilities`、`handles_supporting_tools`、`handles_supporting_stream` 便于路由至具备特定能力的 Provider（`src/client.rs`、`src/provider/retry.rs`）
- 新增 `StreamDecoder` 与 `StreamEvent`，所有 Provider 的 SSE 接口复用同一解析逻辑，确保 `data:` 块、`[DONE]` 终止标记与错误映射的一致性，大幅降低流式实现差异（`src/stream.rs` 及 `src/provider/*/stream.rs`）
- 在共享 `types` 模块内补齐多模态内容、工具调用、流式 Delta 的详尽结构，并引入 `TokenEstimator`/`TokenEstimate`，按 Provider 家族粗略估算请求 token，辅助业务方做配额预检与角色维度的用量拆分（`src/types/mod.rs`）
- `LLMError` 扩展出 `TokenLimitExceeded`、`ModelNotFound`、`InvalidConfig`、`StreamClosed` 等子类，配合各 Provider 的 `parse_*_error` 与 `retry_after_from_headers` 显式提取 `retry_after`、模型 ID、token 超限提示，方便上游区分可重试与不可重试场景（`src/error.rs`、`src/provider/*/error.rs`）

## 0.1.0 - 2025-11-21

- 首次发布统一 LLM 客户端抽象层，包含 `LLMClient` 与 `LLMProvider` trait
- 提供 OpenAI Chat/Responses、Anthropic Messages、Google Gemini Chat 的基础接入实现
- 支持通用 `ChatRequest`、多模态消息、工具调用以及流式输出
- 内建 `ModelConfig` 与 `Credential` 配置映射，便于通过 declarative 方式加载多个 Provider
- 集成 Reqwest 传输适配层与可插拔 transport，使得测试环境可以注入 Mock
