use std::env;

use std::sync::Arc;

use dotenvy::dotenv;
use futures_util::StreamExt;
use kotoba::LLMProvider;
use kotoba::http::reqwest::ReqwestTransport;
use kotoba::provider::openai_chat::OpenAiChatProvider;
use kotoba::types::{ChatOptions, ChatRequest, ContentPart, Message, Role, TextContent};

fn load_env_var(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.trim().is_empty())
}

fn build_request(model: &str) -> ChatRequest {
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
    let _ = dotenv();

    let Some(endpoint) = load_env_var("OPENAI_CHAT_ENDPOINT") else {
        eprintln!("skip live test: OPENAI_CHAT_ENDPOINT missing");
        return;
    };
    let Some(api_key) = load_env_var("OPENAI_CHAT_KEY") else {
        eprintln!("skip live test: OPENAI_CHAT_KEY missing");
        return;
    };
    let Some(model) = load_env_var("OPENAI_CHAT_MODEL") else {
        eprintln!("skip live test: OPENAI_CHAT_MODEL missing");
        return;
    };

    let transport = Arc::new(ReqwestTransport::default());
    let provider = OpenAiChatProvider::new(transport, api_key)
        .with_base_url(endpoint)
        .with_default_model(model.clone());

    let request = build_request(&model);
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
