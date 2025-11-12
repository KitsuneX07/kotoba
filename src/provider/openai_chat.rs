use crate::error::LLMError;
use crate::http::DynHttpTransport;
use crate::provider::{ChatStream, LLMProvider};
use crate::types::{CapabilityDescriptor, ChatRequest, ChatResponse};
use async_trait::async_trait;

/// OpenAI Chat Completions Provider 占位实现 将根据官方文档完善
#[allow(dead_code)]
pub struct OpenAiChatProvider {
    transport: DynHttpTransport,
    base_url: String,
}

impl OpenAiChatProvider {
    /// 创建 OpenAiChatProvider
    pub fn new(transport: DynHttpTransport, base_url: impl Into<String>) -> Self {
        Self {
            transport,
            base_url: base_url.into(),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAiChatProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
        Err(LLMError::NotImplemented {
            feature: "openai_chat::chat",
        })
    }

    async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
        Err(LLMError::NotImplemented {
            feature: "openai_chat::stream_chat",
        })
    }

    fn capabilities(&self) -> CapabilityDescriptor {
        CapabilityDescriptor {
            supports_stream: true,
            supports_image_input: true,
            supports_audio_input: true,
            supports_video_input: false,
            supports_tools: true,
            supports_structured_output: true,
            supports_parallel_tool_calls: true,
        }
    }

    fn name(&self) -> &'static str {
        "openai_chat"
    }
}
