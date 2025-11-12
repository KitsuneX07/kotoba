use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 角色名称，兼容各供应商的 `role` 语义
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Role(pub String);

impl Role {
    pub fn system() -> Self {
        Self("system".to_string())
    }

    pub fn user() -> Self {
        Self("user".to_string())
    }

    pub fn assistant() -> Self {
        Self("assistant".to_string())
    }
}

/// 通用聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// 消息角色
    pub role: Role,
    /// 供应商可选的 name 字段
    pub name: Option<String>,
    /// 多模态内容数组
    #[serde(default)]
    pub content: Vec<ContentPart>,
    /// 附加元数据
    pub metadata: Option<HashMap<String, Value>>,
}

/// 消息内容块，覆盖文本、图像、工具等多模态
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// 文本内容
    Text(TextContent),
    /// 图像内容
    Image(ImageContent),
    /// 音频内容
    Audio(AudioContent),
    /// 视频内容
    Video(VideoContent),
    /// 文件引用
    File(FileContent),
    /// 工具调用
    ToolCall(ToolCall),
    /// 工具结果
    ToolResult(ToolResult),
    /// 未知或供应商自定义内容
    Data { data: Value },
}

/// 文本内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    /// 纯文本
    pub text: String,
}

/// 图像内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    /// 图像来源
    pub source: ImageSource,
    /// OpenAI/Anthropic 等的 detail 配置
    pub detail: Option<ImageDetail>,
    /// 自定义元字段
    pub metadata: Option<HashMap<String, Value>>,
}

/// 图像来源
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// 远程 URL
    Url { url: String },
    /// base64 数据
    Base64 {
        data: String,
        mime_type: Option<String>,
    },
    /// 供应商文件 ID
    FileId { file_id: String },
}

/// 图像 detail
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

/// 音频内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    /// 数据来源
    pub source: MediaSource,
    /// 音频格式
    pub mime_type: Option<String>,
    /// 附加属性
    pub metadata: Option<HashMap<String, Value>>,
}

/// 视频内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoContent {
    /// 数据来源
    pub source: MediaSource,
    /// 视频格式
    pub mime_type: Option<String>,
    /// 附加属性
    pub metadata: Option<HashMap<String, Value>>,
}

/// 文件内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    /// 文件 ID
    pub file_id: String,
    /// 提示供应商文件用途
    pub purpose: Option<String>,
    /// 附加属性
    pub metadata: Option<HashMap<String, Value>>,
}

/// 媒体来源通用定义
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaSource {
    /// base64 inline
    Inline { data: String },
    /// 供应商文件 ID
    FileId { file_id: String },
    /// 远程 URL
    Url { url: String },
}

/// 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// 工具名称
    pub name: String,
    /// 功能描述
    pub description: Option<String>,
    /// 输入 schema（JSON Schema）
    pub input_schema: Option<Value>,
    /// 工具类型
    pub kind: ToolKind,
    /// 附加 provider 定制字段
    pub metadata: Option<HashMap<String, Value>>,
}

/// 工具类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolKind {
    /// 自定义函数
    Function,
    /// 文件检索
    FileSearch,
    /// 网络搜索
    WebSearch,
    /// 计算机操作
    ComputerUse,
    /// 供应商专属工具
    Custom { name: String, config: Option<Value> },
}

/// 工具调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// 工具调用 ID
    pub id: Option<String>,
    /// 工具名称
    pub name: String,
    /// 调用参数
    pub arguments: Value,
    /// 工具类型
    pub kind: ToolCallKind,
}

/// 工具调用类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallKind {
    /// 函数调用
    Function,
    /// 文件搜索
    FileSearch,
    /// 网络搜索
    WebSearch,
    /// 计算机操作
    ComputerUse,
    /// 供应商自定义类型
    Custom { name: String },
}

/// 工具执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// 关联的调用 ID
    pub call_id: Option<String>,
    /// 结果内容
    pub output: Value,
    /// 标记结果是否为错误
    #[serde(default)]
    pub is_error: bool,
    /// 附加内容（如 stdio）
    pub metadata: Option<HashMap<String, Value>>,
}

/// 请求体，兼容不同供应商的公共字段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// 消息列表
    pub messages: Vec<Message>,
    /// 请求配置
    #[serde(default)]
    pub options: ChatOptions,
    /// 可用工具
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// 工具选择策略
    pub tool_choice: Option<ToolChoice>,
    /// 额外响应格式设置
    pub response_format: Option<ResponseFormat>,
    /// 供应商特定元数据
    pub metadata: Option<HashMap<String, Value>>,
}

/// 请求控制参数
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatOptions {
    /// 指定模型名称
    pub model: Option<String>,
    /// 温度
    pub temperature: Option<f32>,
    /// top_p
    pub top_p: Option<f32>,
    /// 最大输出 tokens
    pub max_output_tokens: Option<u32>,
    /// presence_penalty
    pub presence_penalty: Option<f32>,
    /// frequency_penalty
    pub frequency_penalty: Option<f32>,
    /// 并行工具调用
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI/Anthropic reasoning 扩展
    pub reasoning: Option<ReasoningOptions>,
    /// OpenAI service tier、Gemini safety 等自定义参数
    pub extra: HashMap<String, Value>,
}

/// 推理相关配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningOptions {
    /// 推理强度
    pub effort: Option<ReasoningEffort>,
    /// 预算 tokens
    pub budget_tokens: Option<u32>,
    /// 自定义扩展
    pub extra: HashMap<String, Value>,
}

/// 推理强度
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Custom(String),
}

/// 工具选择策略
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// 供应商自动决定
    Auto,
    /// 要求使用任意工具
    Any,
    /// 禁用工具
    None,
    /// 指定工具名称
    Tool { name: String },
    /// 供应商自定义配置
    Custom(Value),
}

/// 响应格式
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// 普通文本
    Text,
    /// JSON 对象
    JsonObject,
    /// JSON Schema
    JsonSchema { schema: Value },
    /// 供应商自定义
    Custom(Value),
}

/// 响应输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// 输出列表（消息/工具等）
    pub outputs: Vec<OutputItem>,
    /// token 用量
    pub usage: Option<TokenUsage>,
    /// 结束原因
    pub finish_reason: Option<FinishReason>,
    /// 实际使用的模型
    pub model: Option<String>,
    /// 供应商元信息
    pub provider: ProviderMetadata,
}

/// 输出项
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputItem {
    /// 完整助手消息
    Message { message: Message, index: usize },
    /// 工具调用
    ToolCall { call: ToolCall, index: usize },
    /// 工具结果
    ToolResult { result: ToolResult, index: usize },
    /// 推理轨迹
    Reasoning { text: String, index: usize },
    /// 供应商特定内容
    Custom { data: Value, index: usize },
}

/// Streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    /// 增量事件
    pub events: Vec<ChatEvent>,
    /// 实时 usage 更新
    pub usage: Option<TokenUsage>,
    /// 是否为最后一个 chunk
    pub is_terminal: bool,
    /// 供应商元信息
    pub provider: ProviderMetadata,
}

/// 流式事件
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatEvent {
    /// 文本增量
    MessageDelta(MessageDelta),
    /// 工具调用增量
    ToolCallDelta(ToolCallDelta),
    /// 工具结果增量
    ToolResultDelta(ToolResultDelta),
    /// 供应商原始事件
    Custom { data: Value },
}

/// 文本增量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// 目标消息索引
    pub index: usize,
    /// 角色
    pub role: Option<Role>,
    /// 内容增量
    pub content: Vec<ContentDelta>,
    /// 结束原因
    pub finish_reason: Option<FinishReason>,
}

/// 内容增量
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// 文本片段
    Text { text: String },
    /// 富文本 JSON
    Json { value: Value },
    /// 工具调用嵌入
    ToolCall { delta: ToolCallDelta },
}

/// 工具调用增量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// 目标索引
    pub index: usize,
    /// 调用 ID
    pub id: Option<String>,
    /// 工具名称
    pub name: Option<String>,
    /// 参数增量
    pub arguments_delta: Option<String>,
    /// 工具类型
    pub kind: Option<ToolCallKind>,
    /// 是否完成
    pub is_finished: bool,
}

/// 工具结果增量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultDelta {
    /// 目标索引
    pub index: usize,
    /// 工具调用 ID
    pub call_id: Option<String>,
    /// 输出增量
    pub output_delta: Option<String>,
    /// 是否为错误
    pub is_error: Option<bool>,
    /// 是否完成
    pub is_finished: bool,
}

/// token 用量
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// prompt tokens
    pub prompt_tokens: Option<u64>,
    /// completion tokens
    pub completion_tokens: Option<u64>,
    /// reasoning tokens
    pub reasoning_tokens: Option<u64>,
    /// 总 tokens
    pub total_tokens: Option<u64>,
    /// 供应商自定义统计
    pub details: Option<HashMap<String, Value>>,
}

/// 结束原因
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
    Error,
    Other(String),
}

/// 供应商元信息
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderMetadata {
    /// 供应商标识（如 openai_chat）
    pub provider: String,
    /// 请求 ID
    pub request_id: Option<String>,
    /// 端点描述
    pub endpoint: Option<String>,
    /// 原始响应片段
    pub raw: Option<Value>,
}

/// 能力描述，方便运行时过滤
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CapabilityDescriptor {
    /// 是否支持流式输出
    pub supports_stream: bool,
    /// 是否支持图像输入
    pub supports_image_input: bool,
    /// 是否支持音频输入
    pub supports_audio_input: bool,
    /// 是否支持视频输入
    pub supports_video_input: bool,
    /// 是否支持工具调用
    pub supports_tools: bool,
    /// 是否支持 JSON/结构化输出
    pub supports_structured_output: bool,
    /// 是否支持多轮函数调用
    pub supports_parallel_tool_calls: bool,
}
