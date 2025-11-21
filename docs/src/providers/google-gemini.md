# Google Gemini

## 适用场景

- 需要一次性发送文本、图片、音频、视频等多模态内容；
- 依赖 Gemini 的 `functionDeclarations` 工具体系、`toolConfig` 调度策略、`generationConfig.response_schema`；
- 需要官方 `:streamGenerateContent?alt=sse` 流式接口。

`capabilities()` 返回：

- `supports_stream = true`
- `supports_image_input = true`
- `supports_audio_input = true`
- `supports_video_input = true`
- `supports_tools = true`
- `supports_structured_output = true`
- `supports_parallel_tool_calls = true`

## 构造方式

```rust
use kotoba_llm::provider::google_gemini::GoogleGeminiProvider;
use kotoba_llm::http::reqwest::default_dyn_transport;

let provider = GoogleGeminiProvider::new(default_dyn_transport()?, std::env::var("GEMINI_KEY")?)
    .with_base_url("https://generativelanguage.googleapis.com")
    .with_default_model("gemini-2.0-flash");
```

- `model` 仅影响 URL（`/v1beta/models/{model}:generateContent`），JSON 体不携带模型字段；
- `normalize_model` 会自动在缺失前缀时补上 `models/`；
- 既支持非流式 `generateContent`，也支持流式 `streamGenerateContent?alt=sse`。

## 请求映射

`build_gemini_body` 主要处理以下逻辑：

1. `system`/`developer` 消息折叠为 `system_instruction`，格式为 `{ role: "system", parts: [{ text }] }`；其余消息进入 `contents` 数组，并将 `role = assistant` 替换为 Gemini 的 `model`。
2. 要求 `contents` 非空，否则返回 “Gemini GenerateContent request requires at least one content message”。
3. `generationConfig`：
   - `temperature`、`top_p`（映射为 `topP`）、`max_output_tokens`（`maxOutputTokens`）、`presence_penalty`、`frequency_penalty`；
   - `response_format = JsonObject` ⇒ `response_mime_type = application/json`；
   - `response_format = JsonSchema` ⇒ 同时写入 `response_schema`；
   - `response_format = Custom` ⇒ 直接把自定义对象作为整个 `generationConfig`，覆盖其他字段。
4. 内容映射：
   - 文本 ⇒ `{ "text": ... }`
   - 图片 ⇒ 
     - Base64 ⇒ `inlineData`（含 `mimeType`）
     - URL/FileId ⇒ `fileData`（`mimeType` 默认为 `application/octet-stream`）
   - 音频/视频 ⇒ 依据 `MediaSource` 生成 `inlineData` 或 `fileData`，并保留 `mimeType`
   - 文件 ⇒ `fileData`
   - 自定义 JSON ⇒ 原样传递
   - `ToolCall`/`ToolResult` 不允许直接放入消息，违例会返回 `LLMError::Validation`
5. 工具：
   - `ToolKind::Function` ⇒ `{ functionDeclarations: [{ name, description, parameters }] }`
   - `ToolKind::Custom` ⇒ 直接使用 `config`，缺省时生成 `{ type: name, name: tool.name }`
   - 其他种类会触发 `LLMError::Validation`
6. `tool_choice` 映射到 `toolConfig.functionCallingConfig`：
   - `Auto` ⇒ 不生成配置，沿用默认自动策略；
   - `Any` ⇒ `{ mode: "any" }`
   - `None` ⇒ `{ mode: "none" }`
   - `Tool { name }` ⇒ `{ mode: "any", allowedFunctionNames: [name] }`
   - `Custom` ⇒ 直接透传
7. `metadata`：HashMap 直接写入。`options.extra`（如 `safetySettings`、`cachedContent`）也完整透传。

## Streaming 与错误

- 流式接口使用 `:streamGenerateContent?alt=sse`，收到的 SSE 由 `create_stream` 解析为 `ChatChunk`；
- 若状态码非 2xx，会先通过 `collect_stream_text` 拼接完整 body，再交给 `parse_gemini_error`，确保错误信息中包含官方字段；
- JSON 解析失败时同样返回 `LLMError::Provider { provider: "google_gemini", ... }`。

## 常见校验

| 问题 | 触发 | 解决 |
| --- | --- | --- |
| 没有用户内容（只有 system） | `contents.is_empty()` | 确保至少推入一条 user/assistant 消息。 |
| 直接把 ToolCall/ToolResult 加入消息 | `convert_content_part` 返回 `LLMError::Validation` | 使用工具定义 + `tool_choice`，由模型触发调用。 |
| 想强制 JSON 输出却只设置 `response_format = Text` | 无自动 JSON | 将 `ResponseFormat::JsonObject` 或 `JsonSchema` 赋值，或在 `options.extra` 自行设置 `generationConfig.response_mime_type`。 |

## extra 建议

- `options.extra["safetySettings"]`：传入 Gemini 官方的安全策略数组；
- `options.extra["cachedContent"]`：绑定到已缓存的上下文；
- `metadata` 可放请求 ID、地理信息等，方便后续分析；
- 若需要细粒度控制工具调度，可直接把完整的 `toolConfig` 结构放在 `ToolChoice::Custom` 中，mapper 会原样传递。
