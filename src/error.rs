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
    /// Signals validation failures in the request payload.
    #[error("invalid request: {message}")]
    Validation { message: String },
    /// Declares that a capability is not supported by the selected provider.
    #[error("feature unsupported: {feature}")]
    UnsupportedFeature { feature: &'static str },
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
