# 更新日志

## 0.1.0 - 2025-11-21

- 首次发布统一 LLM 客户端抽象层，包含 `LLMClient` 与 `LLMProvider` trait
- 提供 OpenAI Chat/Responses、Anthropic Messages、Google Gemini Chat 的基础接入实现
- 支持通用 `ChatRequest`、多模态消息、工具调用以及流式输出
- 内建 `ModelConfig` 与 `Credential` 配置映射，便于通过 declarative 方式加载多个 Provider
- 集成 Reqwest 传输适配层与可插拔 transport，使得测试环境可以注入 Mock
