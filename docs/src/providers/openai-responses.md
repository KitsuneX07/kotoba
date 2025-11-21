# OpenAI Responses

## 适用场景

- 使用 OpenAI Responses API（`/v1/responses`），统一处理文本、多模态、工具与结构化输出；
- 需要 JSON Schema、`include`、`previous_response_id` 等 Responses 专属参数；
- 希望在一个请求中组合 function、文件检索、Web 搜索、Computer Use 等工具。

`capabilities()` 返回：

- `supports_stream = true`
- `supports_image_input = true`
- `supports_audio_input = false`
- `supports_video_input = false`
- `supports_tools = true`
- `supports_structured_output = true`
- `supports_parallel_tool_calls = true`

## 构造方式

与 Chat Provider 一致，提供 `new/with_base_url/with_organization/with_project/with_default_model`。同样地，缺少模型会以 `LLMError::Validation { message: "model is required for OpenAI Responses" }` 结束。

## 请求映射

`build_openai_responses_body` 的关键点：

1. 系统或开发者消息折叠为 `instructions`（以两个换行连接），其余消息进入 `input` 数组，元素类型为：
   ```json
   { "type": "message", "role": "user", "content": [ ... ] }
   ```
2. 输入消息仅允许文本/图像/音频/视频/文件/自定义 JSON；若出现 `ToolCall` 或 `ToolResult`，立即报 `LLMError::Validation`。
3. 图片转换逻辑与 Chat 相同：支持 URL、Base64（拼接为 `data:` URL）、文件 ID，并带 `detail`。
4. 音频 (`input_audio`) 与视频 (`input_video`) 映射与 Chat 相同；代码中已实现，但 `CapabilityDescriptor` 仍标记为 `false`，用于提醒上层谨慎开启。
5. 采样参数：`temperature`、`top_p`、`max_output_tokens`、`parallel_tool_calls`。
6. 推理配置：`ReasoningOptions.effort` 映射到 `reasoning.effort`；`extra` 里的其他字段同样写入 `reasoning` 对象。
7. 工具：
   - `ToolKind::Function` → `{ type: "function", name, description, parameters, strict: true }`；
   - `ToolKind::FileSearch/WebSearch/ComputerUse` → 使用 `metadata` 中的键补全，缺省 `type` 分别为 `file_search`、`web_search_preview`、`computer_use_preview`；
   - `ToolKind::Custom` → 直接透传 `config` 或构造 `{ type: name, name: tool.name }`。
8. `tool_choice` 语义与 Chat 相同，但 Responses 官方用 `"auto"/"required"/"none"`。
9. `response_format` → `text.format`：
   - `Text`：`{ format: { type: "text" } }`
   - `JsonObject`：`{ format: { type: "json_object" } }`
   - `JsonSchema`：自动补上 `name = "response"` 和 `schema`
   - `Custom`：直接把对象作为 `text`
10. `metadata` 与 `options.extra` 直接写入顶层，方便控制 `include`, `service_tier`, `user`, `previous_response_id` 等参数。
11. `stream` = `true/false` 控制 SSE。

## Streaming 与错误

逻辑与 Chat Provider 相同：

- 失败时收集 SSE 文本，调用 `parse_openai_responses_error` 生成 `LLMError`；
- 非流式路径解析 `OpenAiResponsesResponse`，JSON 解析失败同样走 `LLMError::Provider`。

## 工具与结构化输出注意事项

| 功能 | 细节 |
| --- | --- |
| Function 工具 `strict` | 代码默认写入 `strict: true`，若需要关闭可在 `ToolDefinition.metadata` 中加 `{"strict": false}`，映射逻辑会覆盖默认值。 |
| 非 function 工具 | 依赖 `metadata` 补齐字段，例如搜索索引 ID；如果 metadata 为空，则仅提供最基础的 `type`。 |
| 多工具并行 | `ChatOptions.parallel_tool_calls = Some(false)` 会让 `tool_choice` 中的 `disable_parallel_tool_use = true`（在 Responses 默认 `true`），从而限制模型同一轮内只触发一个工具。 |
| JSON 输出 | 推荐显式设置 `response_format = JsonObject/JsonSchema`，否则 Responses 仍可能返回富文本。 |

## 常见校验

- 输入数组为空：因为系统/开发者消息会被折叠，若没有 user/assistant 内容会导致 `input` 缺失，但当前实现允许空输入；建议业务层确保至少有一条用户消息。
- `ToolCall` 直接放在 `Message.content` 中：`convert_input_message` 会立即报错，请将工具相关信息写入 `ToolDefinition` 并交由模型触发。
- 缺失 `max_output_tokens`：Responses API 允许缺省，代码也不会强制要求；如需硬性限制，请在业务层添加校验。

## 配置与 extra

- `extra.organization`、`extra.project` 设置共享 header；
- `ChatOptions.extra` 中可以放 `include`、`service_tier`、`previous_response_id` 等 Responses 特有字段；
- `ChatRequest.metadata` 建议写入 trace id、用户、上下文，用于后续排查。
