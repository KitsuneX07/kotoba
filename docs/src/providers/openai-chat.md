# OpenAI Chat

## 适用场景

- 对接 OpenAI Chat Completions 或兼容代理（如 `https://api.openai.com/v1/chat/completions`）；
- 需要标准 function tool 调用、JSON 模式、流式 SSE；
- 希望在请求体中同时发送文本、图片、音频、视频、文件。

`OpenAiChatProvider` 的 `capabilities()` 返回：

- `supports_stream = true`
- `supports_image_input = true`
- `supports_audio_input = true`
- `supports_video_input = false`（实现已映射视频输入，但出于谨慎暂不对外宣称）
- `supports_tools = true`
- `supports_structured_output = true`
- `supports_parallel_tool_calls = true`

## 构造方式

```rust
use kotoba::provider::openai_chat::OpenAiChatProvider;
use kotoba::http::reqwest::default_dyn_transport;

let transport = default_dyn_transport()?;
let provider = OpenAiChatProvider::new(transport, std::env::var("OPENAI_API_KEY")?)
    .with_base_url("https://api.openai.com")
    .with_organization("org_123")
    .with_project("proj_alpha")
    .with_default_model("gpt-4.1-mini");
```

- 未在 `ChatRequest.options.model` 指定模型时会 fallback 到 `with_default_model`，两者都缺失会抛出 `LLMError::Validation { message: "model is required for OpenAI Chat" }`。
- `organization` 与 `project` 分别映射到 `OpenAI-Organization`、`OpenAI-Project` header。
- `base_url` 自动补全 `/v1/chat/completions`，支持自定义代理。

## 请求映射

`request.rs::build_openai_body` 把 `ChatRequest` 映射到 Chat Completions 结构：

1. `messages`：逐条转换 `Message` → `role/name/content/tool_calls`。工具结果只能出现在 `role = "tool"` 的消息中，且最多一个 `ToolResult`，否则触发 `LLMError::Validation`。
2. `content` 支持：
   - 文本 → `{ type: "text", text }`
   - 图片：URL / Base64（自动拼 `data:mime;base64,...`）/ `file_id`
   - 音频：`input_audio.data/format`
   - 视频：`input_video.source/format`
   - 文件：`file.file_id`
   - 自定义数据：直接嵌入 JSON
3. 采样参数：`temperature`、`top_p`、`max_output_tokens`（映射为 `max_tokens`）、`presence_penalty`、`frequency_penalty`。
4. 工具：仅允许 `ToolKind::Function`；其他种类直接报 `LLMError::Validation`。工具定义含 `name`、`description`、`parameters`。
5. `tool_choice`：`Auto/Any/None` → 字符串，`Tool { name }` → function 对象，`Custom` 完全透传。
6. `response_format`：`Text/JsonObject/JsonSchema/Custom` 全量支持，分别映射到 OpenAI 的 `response_format`。
7. `reasoning`：`ReasoningOptions` 转换为 `reasoning_effort`、`max_reasoning_tokens` 及自定义字段。
8. `metadata`：HashMap → JSON object。
9. `options.extra`：逐键透传，可用于 `service_tier`、`logprobs` 等未统一的参数。
10. `stream`：布尔值控制 SSE。

## Streaming 与错误

- `stream_chat` 使用 `post_json_stream_with_headers` 建立 SSE，成功时返回 `ChatStream`；
- 若 HTTP 状态码非 2xx，会先收集流 body（`collect_stream_text`），再调用 `parse_openai_error` 转换为 `LLMError`；
- 非流式路径会读取完整响应文本再解析 `OpenAiChatResponse`，解析失败统一返回 `LLMError::Provider { provider: "openai_chat", message: ... }`。

## 常见校验

| 触发条件 | 错误 | 排查方式 |
| --- | --- | --- |
| `ChatRequest` 未声明模型且默认模型缺失 | `LLMError::Validation` | 在配置或请求层确保模型名称填写。 |
| `role=tool` 的消息包含多个 `ToolResult` | `LLMError::Validation { message: "tool role expects a single ToolResult content" }` | 将多段工具结果拆分为多条 tool 消息或合并内容。 |
| `ToolCall.kind` 非 `Function` | `LLMError::Validation { message: "OpenAI only supports function tool calls" }` | 统一使用 `ToolKind::Function`，或在业务层做条件判断。 |

## `extra` 建议

- `ChatOptions.extra["service_tier"] = "default" | "scale"`：切换服务等级；
- `ChatRequest.metadata`：可放 trace id、租户信息，方便调试；
- 若代理要求自定义 header，可在构造 `HttpTransport` 时设置。
