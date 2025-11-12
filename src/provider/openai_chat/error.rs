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
