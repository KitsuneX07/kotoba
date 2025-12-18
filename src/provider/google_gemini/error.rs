use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{LLMError, extract_model_identifier, looks_like_token_limit_error};

/// Parses error responses returned by Google Gemini.
pub(crate) fn parse_gemini_error(
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
        code: Option<i32>,
        message: Option<String>,
        status: Option<String>,
        #[allow(dead_code)]
        details: Option<Value>,
    }

    if let Ok(parsed) = serde_json::from_str::<ErrorBody>(body) {
        if let Some(error) = parsed.error {
            let mut message = error.message.unwrap_or_else(|| "unknown error".to_string());
            let status_hint = error.status.as_deref();
            if let Some(status_text) = status_hint {
                if !status_text.is_empty() {
                    message = format!("{message} ({status_text})");
                }
            }

            if looks_like_token_limit_error(status_hint, &message) {
                return LLMError::TokenLimitExceeded {
                    message,
                    estimated: None,
                    limit: None,
                };
            }

            if status == 404 || matches!(status_hint, Some(text) if text == "NOT_FOUND") {
                return LLMError::ModelNotFound {
                    model: extract_model_identifier(&message),
                    message,
                };
            }

            // Combine HTTP status with Google RPC status codes for richer classification.
            return match (status, error.code.unwrap_or(status as i32)) {
                (401, _) => LLMError::Auth { message },
                (403, _) => LLMError::Auth { message },
                (429, _) => LLMError::RateLimit {
                    message,
                    retry_after,
                },
                (400, _) => LLMError::Validation { message },
                (404, _) => LLMError::Provider {
                    provider: "google_gemini",
                    message,
                },
                (code, _) if (500..600).contains(&code) => LLMError::Provider {
                    provider: "google_gemini",
                    message,
                },
                _ => LLMError::Provider {
                    provider: "google_gemini",
                    message,
                },
            };
        }
    }

    // Fallback: if the payload cannot be parsed, return the raw body.
    LLMError::Provider {
        provider: "google_gemini",
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
    "code": 401,
    "message": "API key not valid",
    "status": "UNAUTHENTICATED"
  }
}"#;
        let err = parse_gemini_error(401, body, None);
        match err {
            LLMError::Auth { message } => {
                assert!(message.contains("API key not valid"));
                assert!(message.contains("UNAUTHENTICATED"));
            }
            other => panic!("expected Auth error, got {other:?}"),
        }

        let body = r#"{
  "error": {
    "code": 429,
    "message": "quota exhausted",
    "status": "RESOURCE_EXHAUSTED"
  }
}"#;
        let err = parse_gemini_error(429, body, Some(Duration::from_secs(5)));
        match err {
            LLMError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("quota exhausted"));
                assert!(message.contains("RESOURCE_EXHAUSTED"));
                assert_eq!(retry_after, Some(Duration::from_secs(5)));
            }
            other => panic!("expected RateLimit error, got {other:?}"),
        }
    }

    #[test]
    fn parse_validation_and_provider_errors() {
        let body = r#"{
  "error": {
    "code": 400,
    "message": "Invalid argument: contents",
    "status": "INVALID_ARGUMENT"
  }
}"#;
        let err = parse_gemini_error(400, body, None);
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("Invalid argument"));
                assert!(message.contains("INVALID_ARGUMENT"));
            }
            other => panic!("expected Validation error, got {other:?}"),
        }

        let body = r#"{
  "error": {
    "code": 404,
    "message": "model not found",
    "status": "NOT_FOUND"
  }
}"#;
        let err = parse_gemini_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, message } => {
                assert!(message.contains("model not found"));
                assert!(model.is_none());
            }
            other => panic!("expected ModelNotFound error, got {other:?}"),
        }

        // For non-JSON payloads we expect the fallback Provider branch.
        let body = "not a json";
        let err = parse_gemini_error(500, body, None);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "google_gemini");
                assert!(message.contains("status 500: not a json"));
            }
            other => panic!("expected Provider fallback error, got {other:?}"),
        }
    }

    #[test]
    fn parse_gemini_token_limit_and_model_errors() {
        let body = r#"{
  "error": {
    "code": 400,
    "message": "The prompt tokens exceeded the allowed context window.",
    "status": "INVALID_ARGUMENT"
  }
}"#;
        let err = parse_gemini_error(400, body, None);
        assert!(matches!(err, LLMError::TokenLimitExceeded { .. }));

        let body = r#"{
  "error": {
    "code": 404,
    "message": "Model `gemini-pro-oops` not found.",
    "status": "NOT_FOUND"
  }
}"#;
        let err = parse_gemini_error(404, body, None);
        match err {
            LLMError::ModelNotFound { model, .. } => {
                assert_eq!(model.as_deref(), Some("gemini-pro-oops"));
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }
}
