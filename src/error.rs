use std::time::Duration;

use thiserror::Error;

/// Aggregates every failure mode exposed by the unified LLM client.
///
/// Callers can downcast on the specific variant to decide whether to retry, fall back
/// to another provider, or surface an actionable message to the user interface.
#[derive(Debug, Error)]
pub enum LLMError {
    /// Represents transport-layer or networking failures.
    #[error("transport error: {message}")]
    Transport { message: String },
    /// Reports invalid or missing credentials.
    #[error("auth failure: {message}")]
    Auth { message: String },
    /// Indicates that the provider throttled the request.
    #[error("rate limited: {message}")]
    RateLimit {
        /// Raw message returned by the upstream provider.
        message: String,
        /// Optional wait duration suggested by the provider before retrying.
        retry_after: Option<Duration>,
    },
    /// Indicates that the prompt or expected completion exceeds the allowed token budget.
    #[error("token limit exceeded: {message}")]
    TokenLimitExceeded {
        /// Raw message returned by the provider, kept verbatim for debugging.
        message: String,
        /// Estimated tokens in the request or prompt, if known.
        estimated: Option<usize>,
        /// Reported limit communicated by the provider, if available.
        limit: Option<usize>,
    },
    /// Signals validation failures in the request payload.
    #[error("invalid request: {message}")]
    Validation { message: String },
    /// Declares that a capability is not supported by the selected provider.
    #[error("feature unsupported: {feature}")]
    UnsupportedFeature { feature: &'static str },
    /// Indicates that a requested model or deployment handle could not be resolved.
    #[error("model not found: {message}")]
    ModelNotFound {
        /// Model identifier extracted from the error payload when available.
        model: Option<String>,
        /// Full error message returned by the provider for debugging.
        message: String,
    },
    /// Raised when building or validating configuration fails.
    #[error("invalid configuration for {field}: {reason}")]
    InvalidConfig {
        /// Name of the configuration field that failed validation.
        field: String,
        /// Additional context explaining why the field is invalid.
        reason: String,
    },
    /// Surfaces cancellations triggered explicitly by the caller.
    #[error("request aborted: {message}")]
    Aborted {
        /// Message describing who/what cancelled the request.
        message: String,
    },
    /// Signals that an SSE/streaming channel closed before delivering a DONE marker.
    #[error("stream closed unexpectedly: {message}")]
    StreamClosed {
        /// Provider-supplied or synthetic message describing the closure.
        message: String,
    },
    /// Wraps provider-defined errors that cannot be normalized.
    #[error("provider {provider} error: {message}")]
    Provider {
        /// Name of the provider, such as `openai_chat`.
        provider: &'static str,
        /// Human-readable error message returned by the provider.
        message: String,
    },
    /// Marks functionality that is not yet implemented in the crate.
    #[error("not implemented: {feature}")]
    NotImplemented { feature: &'static str },
    /// Catches opaque or unexpected failures.
    #[error("unknown error: {message}")]
    Unknown { message: String },
}

impl LLMError {
    /// Creates an [`LLMError::Transport`] from a textual description.
    ///
    /// The helper keeps call sites concise and guarantees consistent formatting of
    /// transport failures across the crate.
    ///
    /// # Examples
    ///
    /// ```
    /// use kotoba_llm::error::LLMError;
    ///
    /// let err = LLMError::transport("dns lookup failed");
    /// assert!(matches!(err, LLMError::Transport { .. }));
    /// ```
    pub fn transport<T: Into<String>>(message: T) -> Self {
        Self::Transport {
            message: message.into(),
        }
    }

    /// Creates an [`LLMError::Provider`] with the given provider name and message.
    ///
    /// This helper unifies how provider-specific failures are reported and makes it
    /// trivial to include the provider identifier alongside the message.
    ///
    /// # Examples
    ///
    /// ```
    /// use kotoba_llm::error::LLMError;
    ///
    /// let err = LLMError::provider("openai_chat", "bad JSON payload");
    /// assert!(matches!(err, LLMError::Provider { provider: "openai_chat", .. }));
    /// ```
    pub fn provider<T: Into<String>>(provider: &'static str, message: T) -> Self {
        Self::Provider {
            provider,
            message: message.into(),
        }
    }
}

/// Returns `true` when an error code or message suggests a context/window overflow.
pub(crate) fn looks_like_token_limit_error(code_hint: Option<&str>, message: &str) -> bool {
    if let Some(code) = code_hint {
        let lower = code.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "context_length_exceeded"
                | "max_context_length_exceeded"
                | "prompt_tokens_exceeded"
                | "context_window_exceeded"
        ) || lower.contains("token")
        {
            return true;
        }
    }

    let lower_message = message.to_ascii_lowercase();
    const HINTS: [&str; 6] = [
        "context length",
        "context window",
        "token limit",
        "maximum output tokens",
        "max output tokens",
        "prompt is too long",
    ];
    HINTS.iter().any(|needle| lower_message.contains(needle))
}

/// Attempts to extract a model identifier from an error payload.
pub(crate) fn extract_model_identifier(message: &str) -> Option<String> {
    for delimiter in ['`', '"', '\''] {
        if let Some(value) = between_delimiters(message, delimiter) {
            if !value.trim().is_empty() {
                return Some(value.trim().to_string());
            }
        }
    }
    None
}

fn between_delimiters(message: &str, delimiter: char) -> Option<String> {
    let mut chars = message.char_indices();
    while let Some((start, ch)) = chars.next() {
        if ch == delimiter {
            let start_idx = start + ch.len_utf8();
            if start_idx >= message.len() {
                return None;
            }
            if let Some(rel_end) = message[start_idx..].find(delimiter) {
                let end_idx = start_idx + rel_end;
                return Some(message[start_idx..end_idx].to_string());
            }
        }
    }
    None
}
