# kotoba

[![Crates.io](https://img.shields.io/crates/v/kotoba.svg)](https://crates.io/crates/kotoba)
[![Documentation](https://docs.rs/kotoba/badge.svg)](https://docs.rs/kotoba)
[![Build Status](https://img.shields.io/github/actions/workflow/status/KitsuneX07/kotoba/ci.yml)](https://github.com/KitsuneX07/kotoba/actions)
[![License](https://img.shields.io/crates/l/kotoba.svg)](LICENSE)

**kotoba** 是一个用于 Rust 的统一多厂商 LLM 调用抽象层。它通过一致的类型系统和 trait 抽象，抹平了 OpenAI、Anthropic、Google Gemini 等不同 Provider 之间的 API 差异，为构建 LLM 应用提供类型安全、可扩展的基础设施。

它旨在解决模型碎片化问题，提供统一的消息建模、客户端路由、错误处理与配置加载机制。

## 核心特性

- **统一类型系统 (Unified Type System)**
  提供标准化的 `ChatRequest`, `Message`, `ContentPart` 及 `ChatResponse` 结构。即便是多模态输入、工具调用 (Tool Use) 或推理链 (Reasoning)，也能在统一的结构中处理。

- **Provider 抽象 (Provider Agnostic)**
  基于 `LLMProvider` trait 构建。内置 OpenAI (Chat/Responses), Anthropic Messages, Google Gemini 支持。开发者可轻松扩展自定义的 Provider 网关。

- **灵活的传输层 (Pluggable Transport)**
  网络层与逻辑层解耦。通过 `HttpTransport` 抽象，你可以在生产环境使用 `Reqwest`，在测试环境注入 Mock，或植入自定义的重试与观测中间件。

- **智能路由与调度 (Client-Side Routing)**
  `LLMClient` 支持多 Provider 并存。支持通过 `CapabilityDescriptor` 动态探测能力（如是否支持流式输出、工具调用），实现业务侧的降级策略与 AB 实验。

- **集中式配置 (Declarative Configuration)**
  支持从配置文件批量构建客户端。自动处理 API Key 映射（Bearer/x-api-key）、Base URL 覆写及厂商特有参数注入。

- **完备的错误处理 (Robust Error Handling)**
  细粒度的 `LLMError` 枚举，精确区分网络传输、鉴权失败、速率限制 (Rate Limit) 与模型能力缺失等场景。

## 安装

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
kotoba = "0.1" # 请替换为最新版本
tokio = { version = "1", features = ["full"] }
```

## 🚀 快速开始

下面的示例展示了如何初始化一个 OpenAI Provider 并发送流式对话请求。

```rust
use std::sync::Arc;
use futures_util::StreamExt;
use kotoba::{LLMClient, LLMError};
use kotoba::http::reqwest::default_dyn_transport;
use kotoba::provider::openai_chat::OpenAiChatProvider;
use kotoba::types::{ChatRequest, Message, Role, ContentPart, TextContent};

#[tokio::main]
async fn main() -> Result<(), LLMError> {
    // 1. 初始化 HTTP 传输层与 Provider
    let transport = default_dyn_transport()?;
    let provider = OpenAiChatProvider::new(transport.clone(), std::env::var("OPENAI_API_KEY")?)
        .with_default_model("gpt-4o-mini");

    // 2. 构建客户端并注册 Provider Handle
    let client = LLMClient::builder()
        .register_handle("openai", Arc::new(provider))?
        .build();

    // 3. 构造通用请求
    let request = ChatRequest::new(vec![
        Message {
            role: Role::user(),
            content: vec![ContentPart::Text(TextContent::from("请用一句话介绍 Rust 语言"))],
            ..Default::default()
        }
    ]);

    // 4. 执行流式请求
    let mut stream = client.stream_chat("openai", request).await?;
  
    println!("Response:");
    while let Some(chunk) = stream.next().await {
        if let Ok(c) = chunk {
            // 处理流式增量事件
             println!("Chunk: {:?}", c.events);
        }
    }

    Ok(())
}
```

## 通过配置加载 (Configuration)

在生产环境中，通常需要从动态配置加载多个 Provider。`kotoba` 提供了开箱即用的配置映射能力。

```rust
use kotoba::config::{ModelConfig, ProviderKind, Credential, build_client_from_configs};
use kotoba::http::reqwest::default_dyn_transport;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 模拟加载配置（通常来自 config 文件或配置中心）
    let configs = vec![ModelConfig {
        handle: "gemini".into(),
        provider: ProviderKind::GoogleGemini,
        credential: Credential::ApiKey { 
            header: None, // 自动适配 x-goog-api-key
            key: std::env::var("GEMINI_API_KEY")? 
        },
        default_model: Some("gemini-2.0-flash".into()),
        base_url: None,
        extra: Default::default(),
    }];

    // 一次性构建包含所有 Provider 的客户端
    let client = build_client_from_configs(&configs, default_dyn_transport()?)?;
  
    // 检查能力并调用
    if client.capabilities("gemini").supports_streaming {
        let _ = client.chat("gemini", ChatRequest::from_user_text("Hello Gemini")).await?;
    }

    Ok(())
}
```

## Provider 能力矩阵

`kotoba` 在运行时通过 `CapabilityDescriptor` 暴露各 Provider 的能力差异，由上层业务决定是否禁用或开启特定功能。

| Provider | Streaming | Image Input | Audio Input | Video Input | Tool Use | Structured Output |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenAI Chat** | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **OpenAI Responses** | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ✅ |
| **Anthropic Messages** | ✅ | ✅ (base64) | ❌ | ❌ | ✅ | ⚠️ |
| **Google Gemini** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> **注**：`CapabilityDescriptor` 中标记为 `false` 的能力并不意味着 API 绝对不支持，而是当前 Crate 尚未封装或厂商仅提供受限支持。

## 开发与贡献

欢迎社区贡献新的 Provider 实现或改进现有功能。

### 环境设置
```bash
# 格式化代码
cargo fmt

# 静态检查
cargo clippy

# 运行单元测试 (Request/Response 映射逻辑)
cargo test --lib

# 运行集成测试
cargo test --test integration -- --ignored
```

### 文档
详细的架构说明与 Provider 指南请参阅 `docs/` 目录下的 mdBook 文档。
```bash
mdbook serve docs --open
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
