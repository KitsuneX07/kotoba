use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;

use crate::error::LLMError;
use crate::types::{CapabilityDescriptor, ChatChunk, ChatRequest, ChatResponse};

pub mod openai_chat;

/// 流式响应别名
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatChunk, LLMError>> + Send>>;

/// 统一的 Provider Trait 所有供应商实现该接口即可接入
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// 提交完整请求并等待完整响应
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LLMError>;

    /// 以流式方式返回增量事件
    async fn stream_chat(&self, request: ChatRequest) -> Result<ChatStream, LLMError>;

    /// 描述支持的能力范围
    fn capabilities(&self) -> CapabilityDescriptor;

    /// 供应商名称
    fn name(&self) -> &'static str;
}

/// 线程安全 Provider
pub type DynProvider = Arc<dyn LLMProvider>;
