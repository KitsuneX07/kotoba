use std::collections::HashMap;

use serde_json::Value;

use crate::error::LLMError;
use crate::types::{
    ChatResponse, ContentPart, FinishReason, Message, OutputItem, ProviderMetadata, Role,
    TextContent, TokenUsage, ToolCall, ToolCallKind, ToolResult,
};

use super::types::{GeminiContent, GeminiGenerateContentResponse, GeminiUsageMetadata};

/// 将 Gemini GenerateContentResponse 映射为统一的 ChatResponse
pub(crate) fn map_response(
    resp: GeminiGenerateContentResponse,
    provider: &'static str,
    endpoint: String,
) -> Result<ChatResponse, LLMError> {
    let raw = serde_json::to_value(&resp).ok();

    let mut outputs = Vec::new();
    for (default_index, candidate) in resp.candidates.iter().enumerate() {
        let index = candidate.index.unwrap_or(default_index);
        if let Some(content) = &candidate.content {
            let (message, tool_calls, tool_results) = convert_candidate_content(content, provider)?;
            outputs.push(OutputItem::Message { message, index });
            for call in tool_calls {
                outputs.push(OutputItem::ToolCall { call, index });
            }
            for result in tool_results {
                outputs.push(OutputItem::ToolResult { result, index });
            }
        }
    }

    let finish_reason = resp
        .candidates
        .iter()
        .find_map(|c| c.finish_reason.as_deref().map(convert_finish_reason));
    let usage = resp.usage_metadata.as_ref().map(convert_usage);
    let model = resp.model_version.clone();

    Ok(ChatResponse {
        outputs,
        usage,
        finish_reason,
        model,
        provider: ProviderMetadata {
            provider: provider.to_string(),
            request_id: resp.response_id,
            endpoint: Some(endpoint),
            raw,
        },
    })
}

fn convert_candidate_content(
    content: &GeminiContent,
    provider: &'static str,
) -> Result<(Message, Vec<ToolCall>, Vec<ToolResult>), LLMError> {
    let role = content
        .role
        .as_deref()
        .map(|r| match r {
            // Gemini 使用 model 表示助手
            "model" => Role::assistant(),
            other => Role(other.to_string()),
        })
        .unwrap_or_else(Role::assistant);

    let mut msg_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for part in &content.parts {
        // 函数调用 -> ToolCall 输出项，不放入 message 内容
        if let Some(call) = &part.function_call {
            let tool_call = ToolCall {
                id: None,
                name: call.name.clone(),
                arguments: call.args.clone(),
                kind: ToolCallKind::Function,
            };
            tool_calls.push(tool_call);
            continue;
        }

        // 函数响应 -> ToolResult 输出项，同样不放入 message 内容
        if let Some(resp) = &part.function_response {
            let result = ToolResult {
                call_id: None,
                output: resp.response.clone(),
                is_error: false,
                metadata: None,
            };
            tool_results.push(result);
            continue;
        }

        // 纯文本 part，作为正常消息内容
        if let Some(text) = &part.text {
            if !text.is_empty() {
                msg_parts.push(ContentPart::Text(TextContent { text: text.clone() }));
                continue;
            }
        }

        // 其余多模态 / 元信息整体透传为 Data，供上层按需解析
        let data = serde_json::to_value(part).map_err(|err| LLMError::Provider {
            provider,
            message: format!("failed to serialize Gemini part: {err}"),
        })?;
        msg_parts.push(ContentPart::Data { data });
    }

    let message = Message {
        role,
        name: None,
        content: msg_parts,
        metadata: None,
    };
    Ok((message, tool_calls, tool_results))
}

/// FinishReason 文本 -> 通用 FinishReason
pub(crate) fn convert_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "MALFORMED_FUNCTION_CALL" => FinishReason::FunctionCall,
        // SAFETY / BLOCKLIST / PROHIBITED_CONTENT / SPII / IMAGE_SAFETY 等都视为内容过滤
        "SAFETY" | "RECITATION" | "LANGUAGE" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
        | "IMAGE_SAFETY" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

/// UsageMetadata -> TokenUsage
pub(crate) fn convert_usage(usage: &GeminiUsageMetadata) -> TokenUsage {
    let mut details = HashMap::new();

    if let Some(v) = usage.cached_content_token_count {
        details.insert("cached_content_token_count".to_string(), Value::from(v));
    }
    if let Some(v) = usage.tool_use_prompt_token_count {
        details.insert("tool_use_prompt_token_count".to_string(), Value::from(v));
    }
    if let Some(v) = usage.thoughts_token_count {
        details.insert("thoughts_token_count".to_string(), Value::from(v));
    }
    // 其它细节字段统一透传
    for (k, v) in &usage.extra {
        details.insert(k.clone(), v.clone());
    }

    TokenUsage {
        prompt_tokens: usage.prompt_token_count,
        completion_tokens: usage.candidates_token_count,
        reasoning_tokens: usage.thoughts_token_count,
        total_tokens: usage.total_token_count,
        details: if details.is_empty() {
            None
        } else {
            Some(details)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{GeminiCandidate, GeminiPart, GeminiUsageMetadata};
    use super::*;

    /// 基础非流式响应映射
    #[test]
    fn map_basic_text_response() {
        let candidate = GeminiCandidate {
            content: Some(GeminiContent {
                parts: vec![GeminiPart {
                    text: Some("你好，世界".to_string()),
                    inline_data: None,
                    file_data: None,
                    function_call: None,
                    function_response: None,
                    executable_code: None,
                    code_execution_result: None,
                    extra: HashMap::new(),
                }],
                role: Some("model".to_string()),
                extra: HashMap::new(),
            }),
            finish_reason: Some("STOP".to_string()),
            index: Some(0),
            extra: HashMap::new(),
        };

        let resp = GeminiGenerateContentResponse {
            candidates: vec![candidate],
            prompt_feedback: None,
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(4),
                cached_content_token_count: None,
                candidates_token_count: Some(7),
                total_token_count: Some(11),
                tool_use_prompt_token_count: None,
                thoughts_token_count: None,
                extra: HashMap::new(),
            }),
            model_version: Some("gemini-2.0-flash".to_string()),
            response_id: Some("resp-1".to_string()),
            extra: HashMap::new(),
        };

        let mapped =
            map_response(resp, "google_gemini", "endpoint".to_string()).expect("map succeeds");

        assert_eq!(mapped.model.as_deref(), Some("gemini-2.0-flash"));
        assert!(matches!(mapped.finish_reason, Some(FinishReason::Stop)));
        assert_eq!(mapped.provider.provider, "google_gemini");
        assert_eq!(mapped.provider.endpoint.as_deref(), Some("endpoint"));

        assert_eq!(mapped.outputs.len(), 1);
        match &mapped.outputs[0] {
            OutputItem::Message { message, index } => {
                assert_eq!(*index, 0);
                assert_eq!(message.role.0, "assistant");
                assert_eq!(message.content.len(), 1);
                match &message.content[0] {
                    ContentPart::Text(TextContent { text }) => {
                        assert_eq!(text, "你好，世界");
                    }
                    other => panic!("unexpected content part: {other:?}"),
                }
            }
            other => panic!("unexpected output item: {other:?}"),
        }

        let usage = mapped.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, Some(4));
        assert_eq!(usage.completion_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(11));
    }

    /// FinishReason 映射
    #[test]
    fn convert_finish_reason_variants() {
        assert!(matches!(convert_finish_reason("STOP"), FinishReason::Stop));
        assert!(matches!(
            convert_finish_reason("MAX_TOKENS"),
            FinishReason::Length
        ));
        assert!(matches!(
            convert_finish_reason("MALFORMED_FUNCTION_CALL"),
            FinishReason::FunctionCall
        ));
        assert!(matches!(
            convert_finish_reason("SAFETY"),
            FinishReason::ContentFilter
        ));
        assert!(matches!(
            convert_finish_reason("BLOCKLIST"),
            FinishReason::ContentFilter
        ));
        assert!(matches!(
            convert_finish_reason("OTHER"),
            FinishReason::Other(_)
        ));
    }
}
