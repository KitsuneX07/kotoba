use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;

use super::LLMProvider;
use crate::error::LLMError;
use crate::types::{ChatRequest, ChatResponse};

/// Configuration for retry attempts with exponential backoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryConfig {
    /// Maximum number of retries after the initial attempt.
    pub max_retries: u32,
    /// Initial delay in milliseconds before the first retry.
    pub initial_backoff_ms: u64,
    /// Upper bound applied to the exponential backoff progression.
    pub max_backoff_ms: u64,
    /// Multiplier applied after each retry to grow the delay.
    pub backoff_multiplier: f64,
}

impl RetryConfig {
    /// Creates a new configuration, normalizing obvious misconfigurations.
    pub fn new(
        max_retries: u32,
        initial_backoff_ms: u64,
        max_backoff_ms: u64,
        backoff_multiplier: f64,
    ) -> Self {
        Self {
            max_retries,
            initial_backoff_ms,
            max_backoff_ms: max_backoff_ms.max(initial_backoff_ms),
            backoff_multiplier: if backoff_multiplier < 1.0 {
                1.0
            } else {
                backoff_multiplier
            },
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 250,
            max_backoff_ms: 2_000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Extension trait providing retry helpers for any [`LLMProvider`] implementation.
#[async_trait]
pub trait RetryableLLMProvider: LLMProvider {
    /// Invokes [`LLMProvider::chat`] with exponential backoff when transient failures occur.
    ///
    /// The helper retries up to `config.max_retries` times whenever the provider reports a
    /// [`LLMError::RateLimit`] or [`LLMError::Transport`] variant. When the error includes
    /// a provider-supplied `retry_after`, it takes precedence over the exponential delay.
    async fn chat_with_retry(
        &self,
        request: ChatRequest,
        config: RetryConfig,
    ) -> Result<ChatResponse, LLMError>;
}

#[async_trait]
impl<T> RetryableLLMProvider for T
where
    T: LLMProvider + ?Sized,
{
    async fn chat_with_retry(
        &self,
        request: ChatRequest,
        config: RetryConfig,
    ) -> Result<ChatResponse, LLMError> {
        let mut attempts = 0;
        let mut backoff_ms = config.initial_backoff_ms;
        loop {
            match self.chat(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(err) => {
                    if !err.is_retryable() || attempts >= config.max_retries {
                        return Err(err);
                    }
                    attempts += 1;

                    let delay = retry_delay(&err, backoff_ms);
                    tokio::time::sleep(delay).await;
                    backoff_ms = next_backoff(backoff_ms, config);
                }
            }
        }
    }
}

/// Extracts the `Retry-After` header (in seconds) if present.
///
/// Providers occasionally instruct clients to wait before re-sending requests. When the
/// header is numeric this helper parses it into a [`Duration`]. HTTP-date values are
/// currently ignored because vendors primarily use the numeric form.
pub(crate) fn retry_after_from_headers(headers: &HashMap<String, String>) -> Option<Duration> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, value)| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

fn next_backoff(current_backoff_ms: u64, config: RetryConfig) -> u64 {
    if current_backoff_ms == 0 {
        return 0;
    }
    let multiplier = config.backoff_multiplier.max(1.0);
    let scaled = (current_backoff_ms as f64) * multiplier;
    let clamped = if scaled.is_finite() {
        scaled
    } else {
        f64::from(u32::MAX)
    };
    clamped
        .round()
        .clamp(0.0, config.max_backoff_ms as f64)
        .min(u64::MAX as f64) as u64
}

fn retry_delay(error: &LLMError, fallback_ms: u64) -> Duration {
    match error {
        LLMError::RateLimit {
            retry_after: Some(duration),
            ..
        } => *duration,
        _ => Duration::from_millis(fallback_ms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatStream;
    use crate::types::{CapabilityDescriptor, ChatRequest, ChatResponse, ProviderMetadata};
    use futures_util::stream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Clone, Copy)]
    enum FailureMode {
        RateLimit,
        Auth,
    }

    impl FailureMode {
        fn into_error(self) -> LLMError {
            match self {
                FailureMode::RateLimit => LLMError::RateLimit {
                    message: "throttled".to_string(),
                    retry_after: Some(Duration::from_millis(0)),
                },
                FailureMode::Auth => LLMError::Auth {
                    message: "invalid".to_string(),
                },
            }
        }
    }

    #[derive(Debug)]
    struct TestProvider {
        attempts: Arc<AtomicUsize>,
        succeed_after: Option<usize>,
        failure: FailureMode,
    }

    #[async_trait]
    impl LLMProvider for TestProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LLMError> {
            let current = self.attempts.fetch_add(1, Ordering::SeqCst);
            if let Some(target) = self.succeed_after {
                if current >= target {
                    return Ok(ChatResponse {
                        outputs: Vec::new(),
                        usage: None,
                        finish_reason: None,
                        model: Some("test".to_string()),
                        provider: ProviderMetadata {
                            provider: "test".to_string(),
                            ..Default::default()
                        },
                    });
                }
            }
            Err(self.failure.into_error())
        }

        async fn stream_chat(&self, _request: ChatRequest) -> Result<ChatStream, LLMError> {
            Ok(Box::pin(stream::empty()))
        }

        fn capabilities(&self) -> CapabilityDescriptor {
            CapabilityDescriptor::default()
        }

        fn name(&self) -> &'static str {
            "test"
        }
    }

    fn empty_request() -> ChatRequest {
        ChatRequest {
            messages: Vec::new(),
            options: Default::default(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn chat_with_retry_eventually_succeeds() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = TestProvider {
            attempts: attempts.clone(),
            succeed_after: Some(2),
            failure: FailureMode::RateLimit,
        };

        let response = provider
            .chat_with_retry(
                empty_request(),
                RetryConfig {
                    max_retries: 3,
                    initial_backoff_ms: 0,
                    max_backoff_ms: 0,
                    backoff_multiplier: 2.0,
                },
            )
            .await
            .expect("should succeed after retries");

        assert!(response.model.is_some());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn chat_with_retry_respects_max_retries() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = TestProvider {
            attempts: attempts.clone(),
            succeed_after: None,
            failure: FailureMode::RateLimit,
        };

        let err = provider
            .chat_with_retry(
                empty_request(),
                RetryConfig {
                    max_retries: 2,
                    initial_backoff_ms: 0,
                    max_backoff_ms: 0,
                    backoff_multiplier: 2.0,
                },
            )
            .await
            .expect_err("should exhaust retries");

        assert!(matches!(err, LLMError::RateLimit { .. }));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn chat_with_retry_stops_on_non_retryable_error() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = TestProvider {
            attempts: attempts.clone(),
            succeed_after: None,
            failure: FailureMode::Auth,
        };

        let err = provider
            .chat_with_retry(
                empty_request(),
                RetryConfig {
                    max_retries: 5,
                    initial_backoff_ms: 0,
                    max_backoff_ms: 0,
                    backoff_multiplier: 2.0,
                },
            )
            .await
            .expect_err("should return immediately");

        assert!(matches!(err, LLMError::Auth { .. }));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
