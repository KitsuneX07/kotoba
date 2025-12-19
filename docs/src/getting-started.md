# 快速上手

## 1. 引入依赖

在业务 crate 的 `Cargo.toml` 中通过 crates.io 引入：

```toml
[dependencies]
kotoba-llm = "0.2.0"
```

## 2. 构造 `LLMClient`

`LLMClient` 通过 builder 注册多个 Provider 句柄。下面演示如何以 OpenAI Chat 为例建立同步与流式请求。

```rust
use std::sync::Arc;
use futures_util::StreamExt;
use kotoba_llm::{LLMClient, LLMError};
use kotoba_llm::http::reqwest::default_dyn_transport;
use kotoba_llm::provider::openai_chat::OpenAiChatProvider;
use kotoba_llm::types::{ChatRequest, Message, Role, ContentPart, TextContent};

#[tokio::main]
async fn main() -> Result<(), LLMError> {
    let transport = default_dyn_transport()?;
    let provider = OpenAiChatProvider::new(transport.clone(), std::env::var("OPENAI_API_KEY")?)
        .with_default_model("gpt-4.1-mini");

    let client = LLMClient::builder()
        .register_handle("openai", Arc::new(provider))?
        .build();

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![ContentPart::Text(TextContent {
                text: "请用一句话介绍你自己".into(),
            })],
            metadata: None,
        }],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        metadata: None,
    };

    let response = client.chat("openai", request.clone()).await?;
    println!("同步结果: {:?}", response.outputs);

    let mut stream = client.stream_chat("openai", request).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        // chunk.events 中携带 text delta / tool delta 等统一事件
        println!("delta = {:?}", chunk.events);
    }

    Ok(())
}
```

## 3. 运行期能力筛选

- `client.capabilities(handle)` 返回当前句柄的 `CapabilityDescriptor`，可在运行前检查是否支持流式、多模态或工具。
- `client.handles_supporting_stream()` 与 `client.handles_supporting_tools()` 直接返回满足条件的 handle 列表，便于按能力路由请求。

## 4. 统一配置装载

若需要从配置文件或数据库批量注册 Provider，可使用 `config::build_client_from_configs`：

```rust
use kotoba_llm::config::{ModelConfig, ProviderKind, Credential, build_client_from_configs};
use kotoba_llm::types::{ChatRequest, Message, Role, ContentPart, TextContent};
use kotoba_llm::http::reqwest::default_dyn_transport;

#[tokio::main]
async fn main() -> Result<(), kotoba_llm::LLMError> {
    let configs = vec![ModelConfig {
        handle: "claude".into(),
        provider: ProviderKind::AnthropicMessages,
        credential: Credential::ApiKey { header: None, key: std::env::var("ANTHROPIC_KEY")? },
        default_model: Some("claude-3-5-sonnet".into()),
        base_url: None,
        extra: [
            ("version".into(), serde_json::json!("2023-06-01")),
            ("beta".into(), serde_json::json!("client-tools"))
        ].into_iter().collect(),
        patch: None,
    }];

    let transport = default_dyn_transport()?;
    let client = build_client_from_configs(&configs, transport)?;

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![ContentPart::Text(TextContent { text: "讲一个笑话".into() })],
            metadata: None,
        }],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        metadata: None,
    };

    let _ = client.chat("claude", request).await?;
    Ok(())
}
```
