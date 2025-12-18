//! Unified high-level client abstractions for multi-provider LLM access.
//!
//! This crate defines shared types, transport abstractions, and provider implementations
//! so downstream applications can route chat requests to OpenAI, Anthropic, Google Gemini,
//! or any additional vendor through one cohesive API surface.

pub mod client;
pub mod config;
pub mod error;
pub mod http;
pub mod provider;
pub mod stream;
pub mod types;

pub use client::{LLMClient, LLMClientLike};
pub use error::LLMError;
pub use provider::{ChatStream, LLMProvider};
pub use types::*;
