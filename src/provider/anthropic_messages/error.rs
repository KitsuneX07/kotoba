use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{LLMError, extract_model_identifier, looks_like_token_limit_error};

/// Parses error responses returned by the Anthropic Messages API.
pub(crate) fn parse_anthropic_error(
    status: u16,
    body: &str,
    retry_after: Option<Duration>,
) -> LLMError {
    #[derive(Deserialize)]
    struct ErrorBody {
        error: Option<InnerError>,
    }

    #[derive(Deserialize)]
    struct InnerError {
        message: Option<String>,
        #[allow(dead_code)]
        r#type: Option<String>,
        code: Option<Value>,
    }

    if let Ok(parsed) = serde_json::from_str::<ErrorBody>(body) {
        if let Some(error) = parsed.error {
            let mut message = error.message.unwrap_or_else(|| "unknown error".to_string());
            let code_string = error
                .code
                .as_ref()
                .and_then(|v| v.as_str().map(|s| s.to_string()));
            if let Some(code) = &code_string {
                message = format!("{message} ({code})");
            }
            let code_hint = code_string.as_deref();

            if looks_like_token_limit_error(code_hint, &message) {
                return LLMError::TokenLimitExceeded {
                    message,
                    estimated: None,
                    limit: None,
                };
            }

            if status == 404 || matches!(code_hint, Some(code) if code == "not_found") {
                return LLMError::ModelNotFound {
                    model: extract_model_identifier(&message),
                    message,
                };
            }

            return match status {
                401 | 403 => LLMError::Auth { message },
                429 => LLMError::RateLimit {
                    message,
                    retry_after,
                },
                400 => LLMError::Validation { message },
                code if (500..600).contains(&code) => LLMError::Provider {
                    provider: "anthropic_messages",
                    message,
                },
                _ => LLMError::Provider {
                    provider: "anthropic_messages",
                    message,
                },
            };
        }
    }

    // Fallback: if the payload cannot be parsed, surface the raw body.
    LLMError::Provider {
        provider: "anthropic_messages",
        message: format!("status {status}: {body}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auth_and_rate_limit_errors() {
        let body = r#"{
  "error": {
    "type": "authentication_error",
    "message": "Invalid API key provided",
    "code": "invalid_api_key"
  }
}"#;
        let err = parse_anthropic_error(401, body, None);
        match err {
            LLMError::Auth { message } => {
                assert!(message.contains("Invalid API key provided"));
                assert!(message.contains("invalid_api_key"));
            }
            other => panic!("expected Auth error, got {other:?}"),
        }

        let body = r#"{
  "error": {
    "type": "rate_limit_error",
    "message": "Too many requests",
    "code": "rate_limit_exceeded"
  }
}"#;
        let err = parse_anthropic_error(429, body, Some(Duration::from_secs(2)));
        match err {
            LLMError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("Too many requests"));
                assert!(message.contains("rate_limit_exceeded"));
                assert_eq!(retry_after, Some(Duration::from_secs(2)));
            }
            other => panic!("expected RateLimit error, got {other:?}"),
        }
    }

    #[test]
    fn parse_validation_and_provider_errors() {
        let body = r#"{
  "error": {
    "type": "invalid_request_error",
    "message": "Bad request",
    "code": "invalid_request"
  }
}"#;
        let err = parse_anthropic_error(400, body, None);
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("Bad request"));
                assert!(message.contains("invalid_request"));
            }
            other => panic!("expected Validation error, got {other:?}"),
        }

        let body = "not a json";
        let err = parse_anthropic_error(500, body, None);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "anthropic_messages");
                assert!(message.contains("status 500: not a json"));
            }
            other => panic!("expected Provider fallback error, got {other:?}"),
        }
    }

    #[test]
    fn parse_anthropic_model_not_found() {
        let body = r#"{
  "error": {
    "type": "not_found_error",
    "message": "The model `claude-bogus` was not found",
    "code": "not_found"
  }
}"#;
        let err = parse_anthropic_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, .. } => {
                assert_eq!(model.as_deref(), Some("claude-bogus"));
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn parse_anthropic_token_limit_errors() {
        let body = r#"{
  "error": {
    "type": "invalid_request_error",
    "message": "Request prompt is too long for the context window",
    "code": "context_length_exceeded"
  }
}"#;
        let err = parse_anthropic_error(400, body, None);
        assert!(matches!(err, LLMError::TokenLimitExceeded { .. }));
    }
}
