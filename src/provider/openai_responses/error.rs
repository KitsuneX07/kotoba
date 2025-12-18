use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{LLMError, extract_model_identifier, looks_like_token_limit_error};

/// Parses error responses returned by the OpenAI Responses API.
pub(crate) fn parse_openai_responses_error(
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
                    provider: "openai_responses",
                    message,
                },
                _ => LLMError::Provider {
                    provider: "openai_responses",
                    message,
                },
            };
        }
    }

    LLMError::Provider {
        provider: "openai_responses",
        message: format!("status {status}: {body}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auth_and_rate_limit_errors() {
        let body = r#"{"error":{"message":"invalid api key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        let err = parse_openai_responses_error(401, body, None);
        match err {
            LLMError::Auth { message } => {
                assert!(message.contains("invalid api key"));
                assert!(message.contains("invalid_api_key"));
            }
            other => panic!("expected Auth error, got {other:?}"),
        }

        let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
        let err = parse_openai_responses_error(429, body, Some(Duration::from_secs(1)));
        match err {
            LLMError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("rate limited"));
                assert!(message.contains("rate_limit_exceeded"));
                assert_eq!(retry_after, Some(Duration::from_secs(1)));
            }
            other => panic!("expected RateLimit error, got {other:?}"),
        }
    }

    #[test]
    fn parse_validation_and_generic_provider_errors() {
        let body = r#"{"error":{"message":"bad request","type":"invalid_request_error","code":"some_code"}}"#;
        let err = parse_openai_responses_error(400, body, None);
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("bad request"));
                assert!(message.contains("some_code"));
            }
            other => panic!("expected Validation error, got {other:?}"),
        }

        let body =
            r#"{"error":{"message":"not found","type":"invalid_request_error","code":null}}"#;
        let err = parse_openai_responses_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, message } => {
                assert!(message.contains("not found"));
                assert!(model.is_none());
            }
            other => panic!("expected ModelNotFound error, got {other:?}"),
        }

        // Non-JSON or malformed payloads should fall back to the Provider branch.
        let body = "not a json";
        let err = parse_openai_responses_error(500, body, None);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "openai_responses");
                assert!(message.contains("status 500: not a json"));
            }
            other => panic!("expected Provider fallback error, got {other:?}"),
        }
    }

    #[test]
    fn parse_responses_token_limit_errors() {
        let body = r#"{
  "error": {
    "message": "Input tokens exceed the maximum context window.",
    "type": "invalid_request_error",
    "code": "context_length_exceeded"
  }
}"#;
        let err = parse_openai_responses_error(400, body, None);
        assert!(matches!(err, LLMError::TokenLimitExceeded { .. }));
    }

    #[test]
    fn parse_responses_model_not_found() {
        let body = r#"{
  "error": {
    "message": "The model `responses:gpt-missing` is not available",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}"#;
        let err = parse_openai_responses_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, .. } => {
                assert_eq!(model.as_deref(), Some("responses:gpt-missing"));
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }
}
