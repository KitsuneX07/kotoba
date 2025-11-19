use std::env;
use std::fs;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose};
use dotenvy::dotenv;
use futures_util::StreamExt;
use kotoba::http::reqwest::ReqwestTransport;
use kotoba::provider::openai_chat::OpenAiChatProvider;
use kotoba::types::{
    ChatOptions, ChatRequest, ContentPart, FinishReason, ImageContent, ImageSource, Message, Role,
    TextContent, ToolChoice, ToolDefinition, ToolKind,
};
use kotoba::{LLMProvider, OutputItem};
use serde_json::json;

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_basic_text_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let mut options = ChatOptions::default();
    options.model = Some(model.clone());

    let request = ChatRequest {
        messages: vec![
            Message {
                role: Role("developer".to_string()),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "你是一个有帮助的助手。".to_string(),
                })],
                metadata: None,
            },
            Message {
                role: Role::user(),
                name: None,
                content: vec![ContentPart::Text(TextContent {
                    text: "你好！".to_string(),
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
        .expect("基础文本对话请求应成功");
    let text = first_text_output(&response).expect("助手应返回文本内容");
    assert!(
        text.contains("我"),
        "为了降低不确定性，回答需要包含“我”，实际为：{text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "简单问答应以 stop 结束"
    );
}

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_basic_image_understanding_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let mut options = ChatOptions::default();
    options.model = Some(model.clone());
    options.max_output_tokens = Some(300);

    // 读取本地测试图片并编码为 base64，走 data URL 通道
    let image_bytes = fs::read("tests/assets/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        .expect("test image should be readable");
    let image_b64 = general_purpose::STANDARD.encode(&image_bytes);

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![
                ContentPart::Text(TextContent {
                    text: "这张图片里有什么？".to_string(),
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

    let response = provider.chat(request).await.expect("图像理解请求应成功");
    let text = first_text_output(&response).expect("助手应描述图像内容");
    assert!(
        text.contains("草"),
        "回答需包含“草”方便匹配，实际为：{text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "图像描述通常以 stop 结束"
    );
}

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_basic_tool_call_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let mut options = ChatOptions::default();
    options.model = Some(model.clone());

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![ContentPart::Text(TextContent {
                text: "波士顿今天的天气怎么样？请调用 get_current_weather 工具，并在工具参数中使用 Boston, MA。".to_string(),
            })],
            metadata: None,
        }],
        options,
        tools: vec![ToolDefinition {
            name: "get_current_weather".to_string(),
            description: Some("获取指定位置的当前天气".to_string()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和州，例如 San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            })),
            kind: ToolKind::Function,
            metadata: None,
        }],
        tool_choice: Some(ToolChoice::Tool {
            name: "get_current_weather".to_string(),
        }),
        response_format: None,
        metadata: None,
    };

    let response = provider.chat(request).await.expect("工具调用应成功");
    let tool_call = response.outputs.iter().find_map(|item| {
        if let OutputItem::ToolCall { call, .. } = item {
            Some(call)
        } else {
            None
        }
    });
    assert!(tool_call.is_some(), "模型响应中必须包含工具调用");
    let tool_call = tool_call.expect("已在上方保证存在工具调用");
    let location = tool_call
        .arguments
        .get("location")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    assert!(
        location.contains("Boston"),
        "工具参数应包含 Boston, MA，实际为：{location}"
    );
}

fn build_stream_request(model: &str) -> ChatRequest {
    let mut options = ChatOptions::default();
    options.model = Some(model.to_string());

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

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_live_sync_and_stream() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let request = build_stream_request(&model);
    let response = provider
        .chat(request.clone())
        .await
        .expect("chat request should succeed");
    assert!(
        !response.outputs.is_empty(),
        "chat response should contain outputs"
    );

    let mut stream = provider
        .stream_chat(request)
        .await
        .expect("streaming chat should start");
    let mut saw_chunk = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk should be valid");
        if chunk.is_terminal {
            break;
        }
        if !chunk.events.is_empty() {
            saw_chunk = true;
        }
    }
    assert!(saw_chunk, "stream should yield at least one data chunk");
}

fn build_provider_from_env() -> Option<(OpenAiChatProvider, String)> {
    let Some(endpoint) = load_env_var("OPENAI_CHAT_ENDPOINT") else {
        eprintln!("skip doc example test: OPENAI_CHAT_ENDPOINT missing");
        return None;
    };
    let Some(api_key) = load_env_var("OPENAI_CHAT_KEY") else {
        eprintln!("skip doc example test: OPENAI_CHAT_KEY missing");
        return None;
    };
    let Some(model) = load_env_var("OPENAI_CHAT_MODEL") else {
        eprintln!("skip doc example test: OPENAI_CHAT_MODEL missing");
        return None;
    };

    let transport = Arc::new(ReqwestTransport::default());
    let provider = OpenAiChatProvider::new(transport, api_key)
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
