use std::env;
use std::fs;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose};
use dotenvy::dotenv;
use futures_util::StreamExt;
use kotoba_llm::LLMError;
use kotoba_llm::http::reqwest::ReqwestTransport;
use kotoba_llm::provider::openai_responses::OpenAiResponsesProvider;
use kotoba_llm::types::{
    ChatOptions, ChatRequest, ContentPart, FinishReason, ImageContent, ImageSource, Message, Role,
    TextContent, ToolChoice, ToolDefinition, ToolKind,
};
use kotoba_llm::{LLMProvider, OutputItem};
use serde_json::json;

/// Connectivity test for basic OpenAI Responses text conversations.
#[tokio::test]
#[ignore = "requires valid OpenAI Responses endpoint"]
async fn openai_responses_basic_text_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        ..ChatOptions::default()
    };

    // Mirror the greeting conversation from openai_chat_basic using developer + user roles.
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

    let response = match provider.chat(request).await {
        Ok(resp) => resp,
        Err(LLMError::Auth { message }) => {
            eprintln!("skip openai_responses_basic_text_dialog_live: auth error: {message}");
            // Treat this as an environment issue (quota, etc.) rather than an implementation bug.
            return;
        }
        Err(LLMError::RateLimit { message, .. }) => {
            eprintln!("skip openai_responses_basic_text_dialog_live: rate limit: {message}");
            return;
        }
        Err(LLMError::Transport { message }) => {
            eprintln!("skip openai_responses_basic_text_dialog_live: transport error: {message}");
            return;
        }
        Err(other) => panic!("text response request should succeed: {other:?}"),
    };
    let text = first_text_output(&response).expect("assistant should return text content");
    assert!(
        text.contains("I"),
        "response must contain 'I' to reduce ambiguity; actual: {text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "simple Q&A should end with Stop (status=completed)"
    );
}

/// Connectivity test for OpenAI Responses image understanding.
#[tokio::test]
#[ignore = "requires valid OpenAI Responses endpoint"]
async fn openai_responses_basic_image_understanding_dialog_live() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let options = ChatOptions {
        model: Some(model.clone()),
        max_output_tokens: Some(300),
        ..ChatOptions::default()
    };

    // Read the local test image, encode it as base64, and send via data URL.
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

    let response = match provider.chat(request).await {
        Ok(resp) => resp,
        Err(LLMError::Auth { message }) => {
            eprintln!(
                "skip openai_responses_basic_image_understanding_dialog_live: auth error: {message}"
            );
            return;
        }
        Err(LLMError::RateLimit { message, .. }) => {
            eprintln!(
                "skip openai_responses_basic_image_understanding_dialog_live: rate limit: {message}"
            );
            return;
        }
        Err(LLMError::Transport { message }) => {
            eprintln!(
                "skip openai_responses_basic_image_understanding_dialog_live: transport error: {message}"
            );
            return;
        }
        Err(other) => panic!("image-understanding Responses request should succeed: {other:?}"),
    };
    let text = first_text_output(&response).expect("assistant should describe the image");
    assert!(
        text.contains("grass"),
        "response must mention grass; actual: {text}"
    );
    assert!(
        matches!(response.finish_reason, Some(FinishReason::Stop)),
        "image descriptions typically end with Stop (status=completed)"
    );
}

/// Connectivity test for OpenAI Responses function calls.
#[tokio::test]
#[ignore = "requires valid OpenAI Responses endpoint"]
async fn openai_responses_basic_tool_call_dialog_live() {
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
                text: "What is Boston's weather today? Call get_current_weather with Boston, MA."
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

    let response = match provider.chat(request).await {
        Ok(resp) => resp,
        Err(LLMError::Auth { message }) => {
            eprintln!("skip openai_responses_basic_tool_call_dialog_live: auth error: {message}");
            return;
        }
        Err(LLMError::RateLimit { message, .. }) => {
            eprintln!("skip openai_responses_basic_tool_call_dialog_live: rate limit: {message}");
            return;
        }
        Err(LLMError::Transport { message }) => {
            eprintln!(
                "skip openai_responses_basic_tool_call_dialog_live: transport error: {message}"
            );
            return;
        }
        Err(other) => panic!("Responses function call should succeed: {other:?}"),
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
        "model response must include a function tool call"
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

    // Mirror openai_chat_live_sync_and_stream with English prompts to work across models.
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

/// Connectivity test covering synchronous and streaming Responses calls.
#[tokio::test]
#[ignore = "requires valid OpenAI Responses endpoint"]
async fn openai_responses_live_sync_and_stream() {
    dotenv().ok();
    let Some((provider, model)) = build_provider_from_env() else {
        return;
    };

    let request = build_stream_request(&model);
    let response = match provider.chat(request.clone()).await {
        Ok(resp) => resp,
        Err(LLMError::Auth { message }) => {
            eprintln!("skip openai_responses_live_sync_and_stream (sync): auth error: {message}");
            return;
        }
        Err(LLMError::RateLimit { message, .. }) => {
            eprintln!("skip openai_responses_live_sync_and_stream (sync): rate limit: {message}");
            return;
        }
        Err(LLMError::Transport { message }) => {
            eprintln!(
                "skip openai_responses_live_sync_and_stream (sync): transport error: {message}"
            );
            return;
        }
        Err(other) => panic!("Responses chat request should succeed: {other:?}"),
    };
    assert!(
        !response.outputs.is_empty(),
        "Responses sync call should return at least one output"
    );

    let mut stream = match provider.stream_chat(request).await {
        Ok(stream) => stream,
        Err(LLMError::Auth { message }) => {
            eprintln!("skip openai_responses_live_sync_and_stream (stream): auth error: {message}");
            return;
        }
        Err(LLMError::RateLimit { message, .. }) => {
            eprintln!("skip openai_responses_live_sync_and_stream (stream): rate limit: {message}");
            return;
        }
        Err(LLMError::Transport { message }) => {
            eprintln!(
                "skip openai_responses_live_sync_and_stream (stream): transport error: {message}"
            );
            return;
        }
        Err(other) => panic!("Responses streaming chat should start: {other:?}"),
    };
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
    assert!(
        saw_chunk,
        "Responses streaming interface should emit at least one chunk with events"
    );
}

fn build_provider_from_env() -> Option<(OpenAiResponsesProvider, String)> {
    let Some(endpoint) = load_env_var("OPENAI_RESPONSES_ENDPOINT") else {
        eprintln!("skip doc example test: OPENAI_RESPONSES_ENDPOINT missing");
        return None;
    };
    let Some(api_key) = load_env_var("OPENAI_RESPONSES_KEY") else {
        eprintln!("skip doc example test: OPENAI_RESPONSES_KEY missing");
        return None;
    };
    let Some(model) = load_env_var("OPENAI_RESPONSES_MODEL") else {
        eprintln!("skip doc example test: OPENAI_RESPONSES_MODEL missing");
        return None;
    };

    let transport = Arc::new(ReqwestTransport::default());
    let provider = OpenAiResponsesProvider::new(transport, api_key)
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
