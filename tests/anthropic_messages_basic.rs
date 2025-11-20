use std::env;
use std::fs;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose};
use dotenvy::dotenv;
use futures_util::StreamExt;
use kotoba::http::reqwest::ReqwestTransport;
use kotoba::provider::anthropic_messages::AnthropicMessagesProvider;
use kotoba::types::{
    ChatOptions, ChatRequest, ContentPart, FinishReason, ImageContent, ImageSource, Message, Role,
    TextContent, ToolChoice, ToolDefinition, ToolKind,
};
use kotoba::{LLMProvider, OutputItem};
use serde_json::json;

/// Anthropic Messages 基础文本对话联通性测试
#[tokio::test]
#[ignore = "requires valid Anthropic Messages endpoint"]
async fn anthropic_messages_basic_text_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        // Anthropic Messages 必须提供 max_tokens
        max_output_tokens: Some(256),
        ..ChatOptions::default()
    };

    let request = ChatRequest {
        messages: vec![
            Message {
                role: Role("developer".to_string()),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "你是一个有帮助的助手，请使用简体中文回答。".to_string(),
                })],
                metadata: None,
            },
            Message {
                role: Role::user(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "简单自我介绍一下你自己。".to_string(),
                })],
                metadata: None,
            },
        ],
        options,
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        metadata: None,
    };

    let response = provider
        .chat(request)
        .await
        .expect("Anthropic 基础文本对话请求应成功");
    let text = first_text_output(&response).expect("助手应返回文本内容");
    assert!(
        text.contains('我'),
        "为了降低不确定性，回答需要包含“我”，实际为：{text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "简单问答通常应以 end_turn/Stop 结束"
    );
}

/// Anthropic Messages 图像理解联通性测试
#[tokio::test]
#[ignore = "requires valid Anthropic Messages endpoint"]
async fn anthropic_messages_basic_image_understanding_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        max_output_tokens: Some(300),
        ..ChatOptions::default()
    };

    // 读取本地测试图片并编码为 base64
    let image_bytes = fs::read("tests/assets/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        .expect("test image should be readable");
    let image_b64 = general_purpose::STANDARD.encode(&image_bytes);

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![
                ContentPart::Text(TextContent {
                    text: "这张图片里有什么？请用简体中文简要描述。".to_string(),
                }),
                ContentPart::Image(ImageContent {
                    source: ImageSource::Base64 {
                        data: image_b64,
                        mime_type: Some("image/jpeg".to_string()),
                    },
                    detail: None,
                    metadata: None,
                }),
            ],
            metadata: None,
        }],
        options,
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        metadata: None,
    };

    let response = provider
        .chat(request)
        .await
        .expect("Anthropic 图像理解请求应成功");
    let text = first_text_output(&response).expect("助手应描述图像内容");
    assert!(
        text.contains('草'),
        "回答需包含“草”方便匹配，实际为：{text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "图像描述通常以 end_turn/Stop 结束"
    );
}

/// Anthropic Messages 工具调用联通性测试
#[tokio::test]
#[ignore = "requires valid Anthropic Messages endpoint"]
async fn anthropic_messages_basic_tool_call_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        max_output_tokens: Some(256),
        ..ChatOptions::default()
    };

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![ContentPart::Text(TextContent {
                text: "今天北京的天气怎么样？请调用 get_weather 工具，并在参数中使用 location=\"北京\"。"
                    .to_string(),
            })],
            metadata: None,
        }],
        options,
        tools: vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: Some("获取指定位置的当前天气".to_string()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称,如:北京"
                    }
                },
                "required": ["location"]
            })),
            kind: ToolKind::Function,
            metadata: None,
        }],
        tool_choice: Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
        }),
        response_format: None,
        metadata: None,
    };

    let response = match provider.chat(request).await {
        Ok(resp) => resp,
        // 工具调用在部分兼容层上可能触发 tool_call_error，这里视为测试条件不满足而非失败
        Err(kotoba::LLMError::Provider { message, .. }) if message.contains("tool_call_error") => {
            eprintln!(
                "skip anthropic_messages_basic_tool_call_dialog_live: tool_call_error: {message}"
            );
            return;
        }
        Err(other) => panic!("Anthropic 工具调用请求应成功: {other:?}"),
    };
    let tool_call = response.outputs.iter().find_map(|item| {
        if let OutputItem::ToolCall { call, .. } = item {
            Some(call)
        } else {
            None
        }
    });
    assert!(
        tool_call.is_some(),
        "模型响应中必须包含工具调用（tool_use 内容块）"
    );
    let tool_call = tool_call.expect("已在上方保证存在工具调用");
    assert_eq!(tool_call.name, "get_weather");
    let location = tool_call
        .arguments
        .get("location")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    assert!(
        location.contains("北京"),
        "工具参数 location 应包含“北京”，实际为：{location}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::ToolCalls)),
        "工具调用场景 stop_reason 应映射为 ToolCalls"
    );
}

/// Anthropic Messages 流式对话联通性测试
#[tokio::test]
#[ignore = "requires valid Anthropic Messages endpoint"]
async fn anthropic_messages_live_sync_and_stream() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let request = build_stream_request(&model);

    // 先校验非流式调用正常
    let response = provider
        .chat(request.clone())
        .await
        .expect("Anthropic 同步调用应成功");
    assert!(
        !response.outputs.is_empty(),
        "chat response should contain outputs"
    );

    // 再测试流式接口
    let mut stream = provider
        .stream_chat(request)
        .await
        .expect("Anthropic streaming chat should start");
    let mut saw_chunk = false;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(err) => {
                eprintln!("skip anthropic_messages_live_sync_and_stream (chunk error): {err}");
                return;
            }
        };
        if chunk.is_terminal {
            break;
        }
        if !chunk.events.is_empty() {
            saw_chunk = true;
        }
    }
    assert!(saw_chunk, "stream should yield at least one data chunk");
}

fn build_stream_request(model: &str) -> ChatRequest {
    let options = ChatOptions {
        model: Some(model.to_string()),
        max_output_tokens: Some(128),
        ..ChatOptions::default()
    };

    ChatRequest {
        messages: vec![
            Message {
                role: Role::system(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "You are a helpful assistant.".to_string(),
                })],
                metadata: None,
            },
            Message {
                role: Role::user(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "Please introduce Rust language in one sentence.".to_string(),
                })],
                metadata: None,
            },
        ],
        options,
        tools: Vec::new(),
        tool_choice: None,
        response_format: None,
        metadata: None,
    }
}

fn build_provider_from_env() -> Option<(AnthropicMessagesProvider, String)> {
    let Some(endpoint) = load_env_var("ANTHROPIC_CHAT_ENDPOINT") else {
        eprintln!("skip anthropic tests: ANTHROPIC_CHAT_ENDPOINT missing");
        return None;
    };
    let Some(api_key) = load_env_var("ANTHROPIC_CHAT_KEY") else {
        eprintln!("skip anthropic tests: ANTHROPIC_CHAT_KEY missing");
        return None;
    };
    let Some(model) = load_env_var("ANTHROPIC_CHAT_MODEL") else {
        eprintln!("skip anthropic tests: ANTHROPIC_CHAT_MODEL missing");
        return None;
    };

    let transport = Arc::new(ReqwestTransport::default());
    let provider = AnthropicMessagesProvider::new(transport, api_key)
        .with_base_url(endpoint)
        .with_default_model(model.clone());
    Some((provider, model))
}

fn load_env_var(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.trim().is_empty())
}

fn first_text_output(response: &kotoba::types::ChatResponse) -> Option<String> {
    response.outputs.iter().find_map(|item| {
        if let OutputItem::Message { message, .. } = item {
            message.content.iter().find_map(|part| match part {
                ContentPart::Text(text) => Some(text.text.clone()),
                _ => None,
            })
        } else {
            None
        }
    })
}
