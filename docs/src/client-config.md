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
| `patch` | 可选的 `RequestPatch`，用于在运行时修改请求 URL、Headers 或 Body。详见"请求补丁"章节。 |

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
        patch: None,
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
| 使用 `Credential::ServiceAccount` 配置 Gemini | 当前实现直接拒绝，提示"provider google_gemini does not support service account credential" | 改用 API Key 或在外部服务中交换为 Bearer token 后再注入。 |

## 请求补丁 (Request Patch)

`RequestPatch` 允许你在不修改代码的情况下对 Provider 的 HTTP 请求进行运行时修改，适用于以下场景：

- **添加自定义参数**：向请求 body 添加厂商特有的实验性参数
- **修改请求地址**：覆盖 Provider 的默认 endpoint URL
- **自定义 HTTP Headers**：添加额外的 header 或删除不需要的 header
- **移除不兼容字段**：删除某些 Provider 不支持的字段

### RequestPatch 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `url` | `Option<String>` | 覆盖请求的完整 URL |
| `body` | `Option<Value>` | 深度合并到请求 body 中的 JSON 对象 |
| `headers` | `Option<HashMap<String, Option<String>>>` | 添加/覆盖 HTTP headers。值为 `None` 时删除该 header |
| `remove_fields` | `Option<Vec<String>>` | 按点分路径删除 body 中的字段，支持数组下标 |

### 使用示例

#### 添加自定义参数

```rust
use kotoba_llm::config::{ModelConfig, RequestPatch, ProviderKind, Credential};
use serde_json::json;

let patch = RequestPatch {
    url: None,
    body: Some(json!({
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "user": "kotoba-client"
    })),
    headers: None,
    remove_fields: None,
};

let config = ModelConfig {
    handle: "openai-custom".into(),
    provider: ProviderKind::OpenAiChat,
    credential: Credential::ApiKey {
        header: None,
        key: std::env::var("OPENAI_KEY").unwrap()
    },
    default_model: Some("gpt-4".into()),
    base_url: None,
    extra: std::collections::HashMap::new(),
    patch: Some(patch),
};
```

#### 修改请求地址和添加自定义 Header

```rust
use std::collections::HashMap;

let patch = RequestPatch {
    url: Some("https://my-proxy.local/v1/chat/completions".into()),
    body: None,
    headers: Some([
        ("X-Custom-Header".to_string(), Some("my-value".to_string())),
        ("X-Debug".to_string(), None),  // 删除 X-Debug header
    ].into_iter().collect()),
    remove_fields: None,
};
```

#### 移除不支持的字段

某些自定义 OpenAI 兼容服务可能不支持 `parallel_tool_calls` 参数：

```rust
let patch = RequestPatch {
    url: None,
    body: None,
    headers: None,
    remove_fields: Some(vec![
        "parallel_tool_calls".to_string(),
        "metadata.trace_id".to_string(),  // 支持嵌套路径
    ]),
};
```

#### 综合示例

```rust
let patch = RequestPatch {
    url: Some("https://custom-api.local/chat".into()),
    body: Some(json!({
        "temperature": 0.7,
        "top_p": 0.9,
        "custom_field": {
            "experiment": "A",
            "trace": true
        }
    })),
    headers: Some([
        ("X-Experiment-Id".to_string(), Some("exp-123".to_string())),
        ("X-Old-Header".to_string(), None),
    ].into_iter().collect()),
    remove_fields: Some(vec!["stream_options".to_string()]),
};
```

### 补丁应用顺序

补丁按以下顺序应用到请求中：

1. **URL 替换**：如果 `patch.url` 存在，覆盖原始 URL
2. **Body 深度合并**：将 `patch.body` 递归合并到请求 body 中
3. **Headers 修改**：添加/覆盖/删除 HTTP headers
4. **字段移除**：从 body 中删除指定路径的字段

### 注意事项

- **深度合并**：`body` 中的对象会递归合并，不会简单覆盖整个对象
- **类型替换**：如果 patch 中的值类型与原值不同，会完全替换
- **路径语法**：`remove_fields` 使用点分路径，例如 `metadata.user_id` 或 `messages.0.content`
- **兼容性**：所有 Provider (OpenAI Chat/Responses, Anthropic Messages, Google Gemini) 都支持补丁功能

通过集中配置并统一校验，可以确保业务代码聚焦在 `ChatRequest` 构造与结果处理上，同时降低多环境部署时的差异。
