# 更新日志

## 0.2.0 - 2025-12-19

相较于 0.1.0，此版本主要聚焦在文档与发布体验：

- **全面补齐 Rustdoc**：为 `ModelConfig`、`RequestPatch`、`ChatRequest`、`ChatResponse`、`TokenEstimator` 等核心类型撰写英文注释与示例，实测 `cargo doc` 零警告，便于直接在 docs.rs 阅读。（源文件：`src/config.rs`、`src/types/mod.rs` 等）
- **README 与 mdBook 升级**：同步更新依赖版本、示例代码与能力矩阵，修复 `ChatRequest::new`、`supports_streaming` 等过期 API，确保复制示例即可运行。（`README.md`、`docs/src/getting-started.md`）
- **版本对齐与发布闸门**：所有 crate 元数据与文档统一 bump 至 `0.2.0`；完成 `cargo fmt && cargo clippy --all-targets --all-features && cargo test --all-features && cargo doc --no-deps`，确保发布质量。

## 0.1.0 - 2025-11-21

- 首次发布统一 LLM 客户端抽象层，包含 `LLMClient` 与 `LLMProvider` trait
- 提供 OpenAI Chat/Responses、Anthropic Messages、Google Gemini Chat 的基础接入实现
- 支持通用 `ChatRequest`、多模态消息、工具调用以及流式输出
- 内建 `ModelConfig` 与 `Credential` 配置映射，便于通过 declarative 方式加载多个 Provider
- 集成 Reqwest 传输适配层与可插拔 transport，使得测试环境可以注入 Mock