# Anthropic Messages

## 适用场景

- 对接 Claude 3.x Messages API，既要兼容文本也要兼容 base64 图像输入；
- 需要 `thinking`/reasoning 控制、并行工具开关；
- 需要在请求头中注入 `anthropic-version` 与 `anthropic-beta`。

`capabilities()` 返回：

- `supports_stream = true`
- `supports_image_input = true`（仅 base64 来源）
- `supports_audio_input = false`
- `supports_video_input = false`
- `supports_tools = true`
- `supports_structured_output = false`
- `supports_parallel_tool_calls = true`

## 构造方式

```rust
use kotoba_llm::provider::anthropic_messages::AnthropicMessagesProvider;
use kotoba_llm::http::reqwest::default_dyn_transport;

let provider = AnthropicMessagesProvider::new(default_dyn_transport()?, std::env::var("ANTHROPIC_KEY")?)
    .with_base_url("https://api.anthropic.com")
    .with_version("2023-06-01")
    .with_beta("prompt-caching,claude-3-opus-preview")
    .with_default_model("claude-3-5-sonnet-20241022");
```

- 版本号缺省为 `2023-06-01`，可通过 `extra.version` 覆盖；
- `with_beta` 接受以逗号连接的标记；
- 缺少模型会抛出 `LLMError::Validation`。

## 请求映射

`build_anthropic_body` 的特点：

1. `system` 与 `developer` 角色的文本被折叠为 `system` 字符串（使用两个换行连接），其余消息进入 `messages` 数组，只允许 `user`/`assistant` 两种角色。
2. 要求至少存在一条 `user/assistant` 消息，否则报 “Anthropic Messages request requires at least one user/assistant message”。
3. `ChatOptions.max_output_tokens` 必填，对应 `max_tokens`；缺失会直接报错。
4. 采样：`temperature`、`top_p`。
5. `reasoning`：
   - 如果 `ReasoningOptions.extra` 中包含 `thinking`，将其完整透传；
   - 否则，当 `budget_tokens` 存在时生成 `{ "type": "enabled", "budget_tokens": ... }` 并附加其余 extra；
   - 未提供 `budget_tokens` 时默认不启用 thinking。
6. 工具：
   - `ToolKind::Function` → `{ type: "custom", name, description, input_schema }`；
   - `ToolKind::Custom` → 直接透传 `config`，或在缺省时构造 `{ type: name, name: tool.name }`；
   - 其他种类（如 `FileSearch`）会触发 `LLMError::Validation`。
7. `tool_choice`：`Auto/Any/Tool` 会附带 `disable_parallel_tool_use = !parallel_tool_calls`；`ToolChoice::None` 表示完全不发送 `tool_choice` 字段。
8. 内容支持：
   - 文本；
   - 图像：**仅** Base64 源（URL / 文件 ID 会返回 `LLMError::UnsupportedFeature { feature: "image_source_non_base64" }`）；
   - 工具结果：`ToolResult` 会转成 `tool_result` block，要求 `call_id`（映射为 `tool_use_id`）。
   - 其他内容类型（音频/视频/文件/ToolCall）都会触发 `LLMError::UnsupportedFeature`，如需扩展可通过 `ContentPart::Data` 自定义 JSON。
9. `metadata` 与 `options.extra` 均直接写入顶层。
10. `stream` 布尔值控制 SSE。

## Streaming 与错误

- 非 2xx 响应会将 SSE 全量收集后交由 `parse_anthropic_error`，因此日志中能看到原始 JSON；
- JSON 解析失败时返回 `LLMError::Provider { provider: "anthropic_messages", ... }`；
- `LLMError::UnsupportedFeature` / `Validation` 出现在请求映射阶段，可在单元测试中提前捕获。

## 常见坑位

| 问题 | 触发代码路径 | 定位建议 |
| --- | --- | --- |
| 忘记设置 `max_output_tokens` | `build_anthropic_body` | 在构造 `ChatRequest` 时显式填写，或在配置层为特定模型设置默认值。 |
| 使用非 base64 图像 | `convert_content_part` | 先把远程图片下载转为 Base64，再放入 `ImageContent::Base64`。 |
| `tool_result` 缺少 `call_id` | `convert_content_part` | 执行真实工具后，务必把模型返回的 `ToolCall.id` 回写到 `ToolResult.call_id`。 |
| 未设置 `parallel_tool_calls` 即想禁用并行 | `convert_tool_choice` | 显式把 `ChatOptions.parallel_tool_calls = Some(false)`，生成 `disable_parallel_tool_use = true`。 |

## extra 建议

- `options.extra["stop_sequences"] = ["Observation:"]`：控制 Claude 的停止标记；
- `options.extra["metadata"]`：已经由 `ChatRequest.metadata` 承担，保持语义清晰即可；
- 若要启用自定义 thinking 结构，可直接把完整对象放在 `ReasoningOptions.extra["thinking"]`，代码会跳过自动推导逻辑。
