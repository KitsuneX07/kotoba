use std::env;
use std::fs;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose};
use dotenvy::dotenv;
use futures_util::StreamExt;
use kotoba_llm::http::reqwest::ReqwestTransport;
use kotoba_llm::provider::openai_chat::OpenAiChatProvider;
use kotoba_llm::types::{
    ChatOptions, ChatRequest, ContentPart, FinishReason, ImageContent, ImageSource, Message, Role,
    TextContent, ToolChoice, ToolDefinition, ToolKind,
};
use kotoba_llm::{LLMProvider, OutputItem};
use serde_json::json;

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_basic_text_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        ..ChatOptions::default()
    };

    let request = ChatRequest {
        messages: vec![
            Message {
                role: Role("developer".to_string()),
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
                    text: "Hello there!".to_string(),
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
        .expect("OpenAI Chat text dialog request should succeed");
    let text = first_text_output(&response).expect("assistant should return text content");
    assert!(
        text.contains("I"),
        "response must contain 'I' to reduce ambiguity; actual: {text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "simple Q&A should end with Stop"
    );
}

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_basic_image_understanding_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        max_output_tokens: Some(300),
        ..ChatOptions::default()
    };

    // Read the local fixture and encode it as base64 to build a data URL.
    let image_bytes = fs::read("tests/assets/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
        .expect("test image should be readable");
    let image_b64 = general_purpose::STANDARD.encode(&image_bytes);

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![
                ContentPart::Text(TextContent {
                    text: "What is in this picture?".to_string(),
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
        .expect("image-understanding request should succeed");
    let text = first_text_output(&response).expect("assistant should describe the image");
    assert!(
        text.contains("grass"),
        "response must mention grass; actual: {text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "image descriptions should usually end with Stop"
    );
}

#[tokio::test]
#[ignore = "requires valid OpenAI-compatible endpoint"]
async fn openai_chat_basic_tool_call_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        ..ChatOptions::default()
    };

    let request = ChatRequest {
        messages: vec![Message {
            role: Role::user(),
            name: None,
            content: vec![ContentPart::Text(TextContent {
                text:
                    "What is the weather in Boston today? Call get_current_weather with Boston, MA."
                        .to_string(),
            })],
            metadata: None,
        }],
        options,
        tools: vec![ToolDefinition {
            name: "get_current_weather".to_string(),
            description: Some("Get the current weather for the specified location".to_string()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., San Francisco, CA"
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

    let response = provider
        .chat(request)
        .await
        .expect("tool-call request should succeed");
    let tool_call = response.outputs.iter().find_map(|item| {
        if let OutputItem::ToolCall { call, .. } = item {
            Some(call)
        } else {
            None
        }
    });
    assert!(
        tool_call.is_some(),
        "model response must include a tool call"
    );
    let tool_call = tool_call.expect("tool call should exist per assertion above");
    let location = tool_call
        .arguments
        .get("location")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    assert!(
        location.contains("Boston"),
        "tool argument should contain Boston, MA; actual: {location}"
    );
}

fn build_stream_request(model: &str) -> ChatRequest {
    let options = ChatOptions {
        model: Some(model.to_string()),
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

fn first_text_output(response: &kotoba_llm::types::ChatResponse) -> Option<String> {
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
