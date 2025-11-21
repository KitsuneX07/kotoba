# 概览

kotoba 是一个 Rust 编写的“统一 LLM 调用层”，通过一套共享的 `Message`、`ContentPart`、`ChatRequest`、`ChatResponse` 等类型，将 OpenAI Chat、OpenAI Responses、Anthropic Messages、Google Gemini 等主流厂商封装进统一接口。仓库暴露的 `LLMClient`、`LLMProvider` 与 `HttpTransport` 抽象允许你：

- 在同一进程内注册多个 Provider 句柄，并通过 handle 名称路由请求；
- 复用一致的多模态消息建模，涵盖文本、图片、音频、视频、文件、工具调用以及流式增量事件；
- 在不改变业务代码的情况下切换 HTTP 实现、注入 mock、或扩展自定义 Provider；
- 借助统一错误 `LLMError` 和能力矩阵 `CapabilityDescriptor`，对不同后端进行特性检测和熔断处理。
