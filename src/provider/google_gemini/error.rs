use serde::Deserialize;
use serde_json::Value;

use crate::error::LLMError;

/// 解析 Google Gemini 错误响应
pub(crate) fn parse_gemini_error(status: u16, body: &str) -> LLMError {
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
            if let Some(status_text) = error.status {
                if !status_text.is_empty() {
                    message = format!("{message} ({status_text})");
                }
            }

            // HTTP 状态码 + Google RPC Status 的联合判断
            return match (status, error.code.unwrap_or(status as i32)) {
                (401, _) => LLMError::Auth { message },
                (403, _) => LLMError::Auth { message },
                (429, _) => LLMError::RateLimit {
                    message,
                    retry_after: None,
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

    // 兜底：无法解析为标准错误结构时，直接返回原始 body
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
        let err = parse_gemini_error(401, body);
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
        let err = parse_gemini_error(429, body);
        match err {
            LLMError::RateLimit {
                message,
                retry_after,
            } => {
                assert!(message.contains("quota exhausted"));
                assert!(message.contains("RESOURCE_EXHAUSTED"));
                assert!(retry_after.is_none());
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
        let err = parse_gemini_error(400, body);
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
        let err = parse_gemini_error(404, body);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "google_gemini");
                assert!(message.contains("model not found"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }

        // 非 JSON 或无法解析时，应该走兜底 Provider 分支
        let body = "not a json";
        let err = parse_gemini_error(500, body);
        match err {
            LLMError::Provider { provider, message } => {
                assert_eq!(provider, "google_gemini");
                assert!(message.contains("status 500: not a json"));
            }
            other => panic!("expected Provider fallback error, got {other:?}"),
        }
    }
}
