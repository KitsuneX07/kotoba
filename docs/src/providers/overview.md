# Provider 指南

本章聚焦四个内置 Provider 的特性、差异与使用场景。无论是直接手动构造 Provider，还是通过 `ModelConfig` 装载，都可以参考以下维度挑选：

## 能力矩阵

| Provider | Streaming | 图像输入 | 音频输入 | 视频输入 | 工具调用 | 结构化输出 | 并行工具 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI Chat (`openai_chat`) | ✅ | ✅ | ✅ | ⚠️（`CapabilityDescriptor` 暂未宣称） | ✅ | ✅ | ✅ |
| OpenAI Responses (`openai_responses`) | ✅ | ✅ | ⚠️（暂未宣称） | ⚠️ | ✅（Function/File/Web/Computer） | ✅ | ✅ |
| Anthropic Messages (`anthropic_messages`) | ✅ | ✅（仅支持 base64 图像） | ❌ | ❌ | ✅ | ⚠️（尚未公开 JSON 模式） | ✅ |
| Google Gemini (`google_gemini`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅（含 JSON Schema） | ✅ |

> ⚠️ 表示当前 `CapabilityDescriptor` 中标记为 `false`，即便请求映射支持对应字段，也会谨慎地对外宣告“未正式支持”。

后续章节将深入每个 Provider 的构造、请求映射、Streaming 与调试细节。
