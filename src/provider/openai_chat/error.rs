use serde::Deserialize;
use serde_json::Value;

use crate::error::LLMError;

pub(crate) fn parse_openai_error(status: u16, body: &str) -> LLMError {
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
            if let Some(code) = error.code {
                message = format!("{message} ({code})");
            }
            return match status {
                401 | 403 => LLMError::Auth { message },
                429 => LLMError::RateLimit {
                    message,
                    retry_after: None,
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
        let err = parse_openai_error(401, body);
        match err {
            LLMError::Auth { message } => {
                assert!(message.contains("invalid api key"));
                assert!(message.contains("invalid_api_key"));
            }
            other => panic!("expected Auth error, got {other:?}"),
        }

        let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
        let err = parse_openai_error(429, body);
        match err {
            LLMError::RateLimit { message, retry_after } => {
                assert!(message.contains("rate limited"));
                assert!(message.contains("rate_limit_exceeded"));
                assert!(retry_after.is_none());
            }
            other => panic!("expected RateLimit error, got {other:?}"),
        }
    }

    #[test]
    fn parse_validation_and_generic_provider_errors() {
        let body = r#"{"error":{"message":"bad request","type":"invalid_request_error","code":"some_code"}}"#;
        let err = parse_openai_error(400, body);
        match err {
            LLMError::Validation { message } => {
                assert!(message.contains("bad request"));
                assert!(message.contains("some_code"));
            }
            other => panic!("expected Validation error, got {other:?}"),
        }

        let body = r#"{"error":{"message":"not found","type":"invalid_request_error","code":null}}"#;
        let err = parse_openai_error(404, body);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "openai_chat");
                assert!(message.contains("not found"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }

        // 非 JSON 或无法解析时，应该走兜底 Provider 分支
        let body = "not a json";
        let err = parse_openai_error(500, body);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "openai_chat");
                assert!(message.contains("status 500: not a json"));
            }
            other => panic!("expected Provider fallback error, got {other:?}"),
        }
    }
}
