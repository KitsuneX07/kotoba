use serde_json::Value;

use crate::error::LLMError;
use crate::types::{
    ChatResponse, ContentPart, FinishReason, Message, OutputItem, ProviderMetadata, Role,
    TextContent, TokenUsage, ToolCall, ToolCallKind, ToolResult,
};

use super::types::{OpenAiResponsesResponse, OpenAiResponsesUsage};

pub(crate) fn map_responses_response(
    resp: OpenAiResponsesResponse,
    provider: &'static str,
    endpoint: String,
) -> Result<ChatResponse, LLMError> {
    let raw = serde_json::to_value(&resp).ok();
    let mut outputs = Vec::new();

    for (index, item) in resp.output.iter().enumerate() {
        let kind = item
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        match kind {
            "message" => {
                let message = convert_message_output(item)?;
                outputs.push(OutputItem::Message { message, index });
            }
            "function_call" => {
                let call = convert_function_call_output(item)?;
                outputs.push(OutputItem::ToolCall { call, index });
            }
            "function_call_output" => {
                let result = convert_function_call_result(item)?;
                outputs.push(OutputItem::ToolResult { result, index });
            }
            "reasoning" => {
                if let Some(text) = extract_reasoning_text(item) {
                    outputs.push(OutputItem::Reasoning { text, index });
                } else {
                    outputs.push(OutputItem::Custom {
                        data: item.clone(),
                        index,
                    });
                }
            }
            _ => {
                outputs.push(OutputItem::Custom {
                    data: item.clone(),
                    index,
                });
            }
        }
    }

    let usage = resp.usage.map(convert_usage);
    let finish_reason = convert_finish_reason(resp.status.as_deref(), &resp.error);

    Ok(ChatResponse {
        outputs,
        usage,
        finish_reason,
        model: Some(resp.model),
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: None,
            endpoint: Some(endpoint),
            raw,
        },
    })
}

fn convert_message_output(item: &Value) -> Result<Message, LLMError> {
    let role = item
        .get("role")
        .and_then(|v| v.as_str())
        .map(|s| Role(s.to_string()))
        .unwrap_or_else(|| Role("assistant".to_string()));

    let name = item
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mut content_parts = Vec::new();
    match item.get("content") {
        Some(Value::Array(parts)) => {
            for part in parts {
                let part_type = part
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                match part_type {
                    "output_text" => {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            content_parts.push(ContentPart::Text(TextContent {
                                text: text.to_string(),
                            }));
                        }
                    }
                    _ => {
                        // 对于 refusal / file_citation / url_citation 等，先整体透传为 Data
                        content_parts.push(ContentPart::Data {
                            data: part.clone(),
                        });
                    }
                }
            }
        }
        Some(Value::String(text)) => {
            content_parts.push(ContentPart::Text(TextContent {
                text: text.to_string(),
            }));
        }
        _ => {}
    }

    Ok(Message {
        role,
        name,
        content: content_parts,
        metadata: None,
    })
}

fn convert_function_call_output(item: &Value) -> Result<ToolCall, LLMError> {
    let call_id = item
        .get("call_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let name = item
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let arguments_raw = item
        .get("arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = serde_json::from_str(&arguments_raw).unwrap_or(Value::String(arguments_raw));

    Ok(ToolCall {
        // Chat Completions 将 id 用作后续 tool_result 的关联 ID，这里保持一致，用 call_id 作为泛化后的 id
        id: call_id,
        name,
        arguments,
        kind: ToolCallKind::Function,
    })
}

fn convert_function_call_result(item: &Value) -> Result<ToolResult, LLMError> {
    let call_id = item
        .get("call_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let output_raw = item
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let output = serde_json::from_str(&output_raw).unwrap_or(Value::String(output_raw));

    Ok(ToolResult {
        call_id,
        output,
        is_error: false,
        metadata: None,
    })
}

fn extract_reasoning_text(item: &Value) -> Option<String> {
    let summary = item.get("summary")?;
    let arr = summary.as_array()?;
    let mut buffer = String::new();
    for (idx, entry) in arr.iter().enumerate() {
        if let Some(text) = entry.get("text").and_then(|v| v.as_str()) {
            if idx > 0 {
                buffer.push('\n');
            }
            buffer.push_str(text);
        }
    }
    if buffer.is_empty() {
        None
    } else {
        Some(buffer)
    }
}

fn convert_finish_reason(status: Option<&str>, error: &Option<Value>) -> Option<FinishReason> {
    if let Some(err) = error {
        if !err.is_null() {
            return Some(FinishReason::Error);
        }
    }
    match status {
        Some("completed") => Some(FinishReason::Stop),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

pub(crate) fn convert_usage(usage: OpenAiResponsesUsage) -> TokenUsage {
    let reasoning_tokens = usage
        .output_tokens_details
        .as_ref()
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(|v| v.as_u64());

    TokenUsage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        reasoning_tokens,
        total_tokens: usage.total_tokens,
        details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn map_text_only_response() {
        let resp = OpenAiResponsesResponse {
            id: "resp_1".to_string(),
            object: "response".to_string(),
            created_at: Some(1),
            status: Some("completed".to_string()),
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-4.1".to_string(),
            output: vec![json!({
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello responses",
                        "annotations": []
                    }
                ]
            })],
            parallel_tool_calls: Some(true),
            previous_response_id: None,
            reasoning: None,
            store: Some(true),
            temperature: Some(1.0),
            text: Some(json!({"format": {"type": "text"}})),
            tool_choice: Some(json!("auto")),
            tools: Some(Vec::new()),
            top_p: Some(1.0),
            truncation: Some("disabled".to_string()),
            usage: Some(OpenAiResponsesUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                total_tokens: Some(15),
                input_tokens_details: None,
                output_tokens_details: Some(json!({"reasoning_tokens": 0})),
            }),
            user: None,
            metadata: None,
        };

        let mapped = map_responses_response(resp, "openai_responses", "endpoint".into())
            .expect("map_responses_response should succeed");

        assert_eq!(mapped.model.as_deref(), Some("gpt-4.1"));
        assert!(matches!(mapped.finish_reason, Some(FinishReason::Stop)));
        assert_eq!(mapped.provider.provider, "openai_responses");
        assert_eq!(mapped.provider.endpoint.as_deref(), Some("endpoint"));

        // 有一个 Message 输出
        assert_eq!(mapped.outputs.len(), 1);
        match &mapped.outputs[0] {
            OutputItem::Message { message, index } => {
                assert_eq!(*index, 0);
                assert_eq!(message.role.0, "assistant");
                assert_eq!(message.content.len(), 1);
                match &message.content[0] {
                    ContentPart::Text(TextContent { text }) => {
                        assert_eq!(text, "hello responses");
                    }
                    other => panic!("unexpected content part: {other:?}"),
                }
            }
            other => panic!("unexpected output item: {other:?}"),
        }

        let usage = mapped.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
        assert_eq!(usage.reasoning_tokens, Some(0));
    }

    #[test]
    fn map_function_call_and_result() {
        let resp = OpenAiResponsesResponse {
            id: "resp_2".to_string(),
            object: "response".to_string(),
            created_at: None,
            status: Some("completed".to_string()),
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-4.1".to_string(),
            output: vec![
                json!({
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_current_weather",
                    "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}",
                    "status": "completed"
                }),
                json!({
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_1",
                    "output": "{\"temperature\": 25}",
                    "status": "completed"
                }),
            ],
            parallel_tool_calls: Some(true),
            previous_response_id: None,
            reasoning: None,
            store: Some(true),
            temperature: Some(1.0),
            text: None,
            tool_choice: Some(json!("auto")),
            tools: None,
            top_p: Some(1.0),
            truncation: Some("disabled".to_string()),
            usage: None,
            user: None,
            metadata: None,
        };

        let mapped = map_responses_response(resp, "openai_responses", "endpoint".into())
            .expect("map_responses_response should succeed");

        assert_eq!(mapped.outputs.len(), 2);
        let mut saw_call = false;
        let mut saw_result = false;
        for item in &mapped.outputs {
            match item {
                OutputItem::ToolCall { call, index } => {
                    saw_call = true;
                    assert_eq!(*index, 0);
                    assert_eq!(call.id.as_deref(), Some("call_1"));
                    assert_eq!(call.name, "get_current_weather");
                    assert_eq!(
                        call.arguments["location"],
                        json!("Boston, MA")
                    );
                    assert_eq!(call.kind, ToolCallKind::Function);
                }
                OutputItem::ToolResult { result, index } => {
                    saw_result = true;
                    assert_eq!(*index, 1);
                    assert_eq!(result.call_id.as_deref(), Some("call_1"));
                    assert_eq!(result.output["temperature"], json!(25));
                }
                _ => {}
            }
        }
        assert!(saw_call && saw_result);
    }
}
