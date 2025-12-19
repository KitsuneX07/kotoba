use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{LLMError, extract_model_identifier, looks_like_token_limit_error};

pub(crate) fn parse_openai_error(
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
                .and_then(|value| value.as_str().map(|s| s.to_string()))
                .or_else(|| error.code.clone().map(|value| value.to_string()));
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

            if status == 404 || matches!(code_hint, Some(code) if code == "model_not_found") {
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
                404 => LLMError::Provider {
                    provider: "openai_chat",
                    message,
                },
                _ => LLMError::Provider {
                    provider: "openai_chat",
                    message,
                },
            };
        }
    }
    LLMError::Provider {
        provider: "openai_chat",
        message: format!("status {status}: {body}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auth_and_rate_limit_errors() {
        let body = r#"{"error":{"message":"invalid api key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        let err = parse_openai_error(401, body, None);
        match err {
            LLMError::Auth { message } => {
                assert!(message.contains("invalid api key"));
                assert!(message.contains("invalid_api_key"));
            }
            other => panic!("expected Auth error, got {other:?}"),
        }

        let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
        let err = parse_openai_error(429, body, Some(Duration::from_secs(3)));
        match err {
            LLMError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("rate limited"));
                assert!(message.contains("rate_limit_exceeded"));
                assert_eq!(retry_after, Some(Duration::from_secs(3)));
            }
            other => panic!("expected RateLimit error, got {other:?}"),
        }
    }

    #[test]
    fn parse_validation_and_generic_provider_errors() {
        let body = r#"{"error":{"message":"bad request","type":"invalid_request_error","code":"some_code"}}"#;
        let err = parse_openai_error(400, body, None);
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("bad request"));
                assert!(message.contains("some_code"));
            }
            other => panic!("expected Validation error, got {other:?}"),
        }

        let body =
            r#"{"error":{"message":"not found","type":"invalid_request_error","code":null}}"#;
        let err = parse_openai_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, message } => {
                assert!(message.contains("not found"));
                assert!(model.is_none());
            }
            other => panic!("expected ModelNotFound error, got {other:?}"),
        }

        // Non-JSON or unparseable responses should fall back to the Provider branch.
        let body = "not a json";
        let err = parse_openai_error(500, body, None);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "openai_chat");
                assert!(message.contains("status 500: not a json"));
            }
            other => panic!("expected Provider fallback error, got {other:?}"),
        }
    }

    #[test]
    fn parse_token_limit_errors() {
        let body = r#"{
  "error": {
    "message": "This model's maximum context length is 8192 tokens. However, you requested 12000 tokens.",
    "type": "invalid_request_error",
    "code": "context_length_exceeded"
  }
}"#;
        let err = parse_openai_error(400, body, None);
        match err {
            LLMError::TokenLimitExceeded {
                message,
                estimated,
                limit,
            } => {
                assert!(message.contains("maximum context length"));
                assert!(estimated.is_none());
                assert!(limit.is_none());
            }
            other => panic!("expected TokenLimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn parse_model_not_found_errors() {
        let body = r#"{
  "error": {
    "message": "The model `gpt-unknown` does not exist",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}"#;
        let err = parse_openai_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, message } => {
                assert_eq!(model.as_deref(), Some("gpt-unknown"));
                assert!(message.contains("model_not_found"));
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }
}
