# 客户端与配置装载

`config::build_client_from_configs` 让你在启动阶段就把所有 Provider 注册好，避免在业务代码里散落大量构造逻辑。本章说明配置字段及各 Provider 支持的额外键。

## ModelConfig 字段

| 字段 | 说明 |
| --- | --- |
| `handle` | 注册到 `LLMClient` 的唯一名称，后续 `client.chat(handle, ..)` 通过它路由。重复 handle 会立即触发 `LLMError::Validation`。 |
| `provider` | `ProviderKind` 枚举，当前支持 `OpenAiChat`、`OpenAiResponses`、`AnthropicMessages`、`GoogleGemini`。 |
| `credential` | `Credential::ApiKey { header, key }`、`Credential::Bearer { token }`、`Credential::ServiceAccount { json }`、`Credential::None`。除 `ServiceAccount`/`None` 外都会映射到对应 Provider；不满足条件时返回 `LLMError::Auth`。 |
| `default_model` | 当 `ChatRequest.options.model` 为空时的兜底模型。绝大多数 Provider 都在请求阶段要求模型，缺失会报 `LLMError::Validation`。 |
| `base_url` | 可选的自定义地址，便于本地代理或企业网关。构造时会调用 Provider 的 `with_base_url`。 |
| `extra` | HashMap<String, Value>，按 Provider 约定解析。未知键会被忽略。 |

## Credential 注意事项

- OpenAI 与 Anthropic/Gemini 均要求 API Key 或 Bearer Token；`Credential::None` 会触发 `LLMError::Auth`。
- `Credential::ServiceAccount` 当前尚未被任何 Provider 支持，会直接返回 `LLMError::Auth`，以免让调用者误以为可以使用 JSON 凭证。
- 可以用 `header` 字段覆盖默认 header 名，例如某些代理要求 `X-API-Key`。若为空，代码会按 Provider 既定 header（`Authorization`, `x-api-key`, `x-goog-api-key` 等）填写。

## extra 字段约定

| Provider | `extra` 键 | 作用 |
| --- | --- | --- |
| OpenAiChat / OpenAiResponses | `organization`、`project` | 分别映射到 `OpenAI-Organization`、`OpenAI-Project` header。 |
| AnthropicMessages | `version`、`beta` | 映射到 `anthropic-version` 与 `anthropic-beta` header，用逗号分隔多个 beta。 |
| GoogleGemini | （暂无默认键） | 可以自定义，例如 `safetySettings`、`cachedContent`，直接透传到请求 JSON。 |

## 使用示例

```rust
use kotoba_llm::config::{ModelConfig, ProviderKind, Credential, build_client_from_configs};
use kotoba_llm::http::reqwest::default_dyn_transport;

fn load_client() -> Result<kotoba_llm::LLMClient, kotoba_llm::LLMError> {
    let mut extras = std::collections::HashMap::new();
    extras.insert("organization".to_string(), serde_json::json!("org_123"));

    let configs = vec![ModelConfig {
        handle: "openai-primary".into(),
        provider: ProviderKind::OpenAiChat,
        credential: Credential::ApiKey { header: None, key: std::env::var("OPENAI_KEY")? },
        default_model: Some("gpt-4.1-mini".into()),
        base_url: None,
        extra: extras,
    }];

    let transport = default_dyn_transport()?;
    build_client_from_configs(&configs, transport)
}
```

## 常见错误与防御

| 问题 | 表现 | 解决办法 |
| --- | --- | --- |
| 忘记在配置里设置 `default_model` | Provider 构造成功，但在运行时调用 `chat` 会因缺少模型返回 `LLMError::Validation` | 在配置层约束必须指定 `default_model`，或在业务层始终给 `ChatRequest.options.model` 赋值。 |
| `extra` 键误拼写 | 不会报错，但相应 header/字段不会生效 | 在配置 schema 或集成测试中校验常见键是否存在。 |
| 同一个 handle 重复出现在配置里 | `build_client_from_configs` 会直接报 `LLMError::Validation { message: "duplicate model handle: ..." }` | 在生成配置时先做去重，或按照 Provider 目的命名（如 `openai-fallback`）。 |
| 使用 `Credential::ServiceAccount` 配置 Gemini | 当前实现直接拒绝，提示“provider google_gemini does not support service account credential” | 改用 API Key 或在外部服务中交换为 Bearer token 后再注入。 |

通过集中配置并统一校验，可以确保业务代码聚焦在 `ChatRequest` 构造与结果处理上，同时降低多环境部署时的差异。
