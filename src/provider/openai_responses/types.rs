use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 非流式 Responses 主体
#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiResponsesResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    #[serde(default)]
    pub(crate) created_at: Option<u64>,
    #[serde(default)]
    pub(crate) status: Option<String>,
    #[serde(default)]
    pub(crate) error: Option<Value>,
    #[serde(default)]
    pub(crate) incomplete_details: Option<Value>,
    #[serde(default)]
    pub(crate) instructions: Option<String>,
    #[serde(default)]
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) output: Vec<Value>,
    #[serde(default)]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub(crate) previous_response_id: Option<String>,
    #[serde(default)]
    pub(crate) reasoning: Option<Value>,
    #[serde(default)]
    pub(crate) store: Option<bool>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) text: Option<Value>,
    #[serde(default)]
    pub(crate) tool_choice: Option<Value>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<Value>>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(default)]
    pub(crate) truncation: Option<String>,
    #[serde(default)]
    pub(crate) usage: Option<OpenAiResponsesUsage>,
    #[serde(default)]
    pub(crate) user: Option<String>,
    #[serde(default)]
    pub(crate) metadata: Option<Value>,
}

/// usage 结构
#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiResponsesUsage {
    #[serde(default)]
    pub(crate) input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) input_tokens_details: Option<Value>,
    #[serde(default)]
    pub(crate) output_tokens_details: Option<Value>,
}

/// SSE 事件 envelope
#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct OpenAiResponsesStreamEvent {
    #[serde(rename = "type")]
    pub(crate) event_type: String,
    #[serde(default)]
    pub(crate) response: Option<OpenAiResponsesResponse>,
    #[serde(default)]
    pub(crate) output_index: Option<usize>,
    #[serde(default)]
    pub(crate) content_index: Option<usize>,
    #[serde(default)]
    pub(crate) delta: Option<String>,
    #[serde(default)]
    pub(crate) item_id: Option<String>,
    #[serde(default)]
    pub(crate) item: Option<Value>,
    #[serde(default)]
    pub(crate) part: Option<Value>,
}
