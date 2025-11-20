use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::LLMClient;
use crate::error::LLMError;
use crate::http::DynHttpTransport;
use crate::provider::DynProvider;
use crate::provider::anthropic_messages::AnthropicMessagesProvider;
use crate::provider::google_gemini::GoogleGeminiProvider;
use crate::provider::openai_chat::OpenAiChatProvider;
use crate::provider::openai_responses::OpenAiResponsesProvider;

/// 模型配置 描述一个可调用后端
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 自定义句柄 例如 `default-openai`
    pub handle: String,
    pub provider: ProviderKind,
    pub credential: Credential,
    pub default_model: Option<String>,
    pub base_url: Option<String>,
    /// 附加设置 例如 service_tier 或 safetySettings
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

/// 供应商类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GoogleGemini,
}

/// 鉴权信息
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    /// 简单 API Key
    ApiKey {
        /// header 名称 留空时按 provider 默认
        header: Option<String>,
        /// 密钥
        key: String,
    },
    /// Bearer Token
    Bearer { token: String },
    /// Google/GCP Service Account JSON
    ServiceAccount { json: Value },
    /// 无需鉴权的本地 provider
    None,
}

/// 根据一组模型配置构建 LLMClient
pub fn build_client_from_configs(
    configs: &[ModelConfig],
    transport: DynHttpTransport,
) -> Result<LLMClient, LLMError> {
    let mut builder = LLMClient::builder();

    for config in configs {
        let provider = build_provider_from_config(config, transport.clone())?;
        builder = builder.register_handle(config.handle.clone(), provider);
    }

    Ok(builder.build())
}

fn build_provider_from_config(
    config: &ModelConfig,
    transport: DynHttpTransport,
) -> Result<DynProvider, LLMError> {
    let provider: DynProvider = match config.provider {
        ProviderKind::OpenAiChat => {
            let api_key = extract_api_key(&config.credential, "openai_chat")?;
            let mut provider = OpenAiChatProvider::new(transport, api_key);

            if let Some(base_url) = &config.base_url {
                provider = provider.with_base_url(base_url.clone());
            }
            if let Some(model) = &config.default_model {
                provider = provider.with_default_model(model.clone());
            }

            if let Some(Value::String(org)) = config.extra.get("organization") {
                provider = provider.with_organization(org.clone());
            }
            if let Some(Value::String(project)) = config.extra.get("project") {
                provider = provider.with_project(project.clone());
            }

            Arc::new(provider)
        }
        ProviderKind::OpenAiResponses => {
            let api_key = extract_api_key(&config.credential, "openai_responses")?;
            let mut provider = OpenAiResponsesProvider::new(transport, api_key);

            if let Some(base_url) = &config.base_url {
                provider = provider.with_base_url(base_url.clone());
            }
            if let Some(model) = &config.default_model {
                provider = provider.with_default_model(model.clone());
            }

            if let Some(Value::String(org)) = config.extra.get("organization") {
                provider = provider.with_organization(org.clone());
            }
            if let Some(Value::String(project)) = config.extra.get("project") {
                provider = provider.with_project(project.clone());
            }

            Arc::new(provider)
        }
        ProviderKind::AnthropicMessages => {
            let api_key = extract_api_key(&config.credential, "anthropic_messages")?;
            let mut provider = AnthropicMessagesProvider::new(transport, api_key);

            if let Some(base_url) = &config.base_url {
                provider = provider.with_base_url(base_url.clone());
            }
            if let Some(model) = &config.default_model {
                provider = provider.with_default_model(model.clone());
            }

            if let Some(Value::String(version)) = config.extra.get("version") {
                provider = provider.with_version(version.clone());
            }
            if let Some(Value::String(beta)) = config.extra.get("beta") {
                provider = provider.with_beta(beta.clone());
            }

            Arc::new(provider)
        }
        ProviderKind::GoogleGemini => {
            let api_key = extract_api_key(&config.credential, "google_gemini")?;
            let mut provider = GoogleGeminiProvider::new(transport, api_key);

            if let Some(base_url) = &config.base_url {
                provider = provider.with_base_url(base_url.clone());
            }
            if let Some(model) = &config.default_model {
                provider = provider.with_default_model(model.clone());
            }

            Arc::new(provider)
        }
    };

    Ok(provider)
}

fn extract_api_key(credential: &Credential, provider: &'static str) -> Result<String, LLMError> {
    match credential {
        Credential::ApiKey { key, .. } => Ok(key.clone()),
        Credential::Bearer { token } => Ok(token.clone()),
        Credential::ServiceAccount { .. } => Err(LLMError::Auth {
            message: format!("provider {provider} does not support service account credential"),
        }),
        Credential::None => Err(LLMError::Auth {
            message: format!("provider {provider} requires credential"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::reqwest::default_dyn_transport;

    /// 验证所有 ProviderKind 分支都可以被构建并注册到 LLMClient
    #[test]
    fn build_client_from_configs_supports_all_providers() {
        let transport = default_dyn_transport().expect("transport");

        let configs = vec![
            ModelConfig {
                handle: "openai-chat".to_string(),
                provider: ProviderKind::OpenAiChat,
                credential: Credential::ApiKey {
                    header: None,
                    key: "test-key-chat".to_string(),
                },
                default_model: Some("gpt-4.1-mini".to_string()),
                base_url: None,
                extra: HashMap::new(),
            },
            ModelConfig {
                handle: "openai-responses".to_string(),
                provider: ProviderKind::OpenAiResponses,
                credential: Credential::ApiKey {
                    header: None,
                    key: "test-key-responses".to_string(),
                },
                default_model: Some("gpt-4.1-mini".to_string()),
                base_url: None,
                extra: HashMap::new(),
            },
            ModelConfig {
                handle: "anthropic-messages".to_string(),
                provider: ProviderKind::AnthropicMessages,
                credential: Credential::ApiKey {
                    header: None,
                    key: "test-key-anthropic".to_string(),
                },
                default_model: Some("claude-3-5-sonnet".to_string()),
                base_url: None,
                extra: HashMap::new(),
            },
            ModelConfig {
                handle: "gemini-generate".to_string(),
                provider: ProviderKind::GoogleGemini,
                credential: Credential::ApiKey {
                    header: None,
                    key: "test-key-gemini".to_string(),
                },
                default_model: Some("gemini-2.0-flash".to_string()),
                base_url: None,
                extra: HashMap::new(),
            },
        ];

        let client = build_client_from_configs(&configs, transport).expect("client");
        let mut handles = client.handles();
        handles.sort();

        assert_eq!(
            handles,
            vec![
                "anthropic-messages".to_string(),
                "gemini-generate".to_string(),
                "openai-chat".to_string(),
                "openai-responses".to_string(),
            ]
        );
    }

    #[test]
    fn build_client_from_configs_builds_handles() {
        let transport = default_dyn_transport().expect("transport");

        let configs = vec![
            ModelConfig {
                handle: "openai-default".to_string(),
                provider: ProviderKind::OpenAiChat,
                credential: Credential::ApiKey {
                    header: None,
                    key: "test-key".to_string(),
                },
                default_model: Some("gpt-4.1-mini".to_string()),
                base_url: None,
                extra: HashMap::new(),
            },
            ModelConfig {
                handle: "gemini-default".to_string(),
                provider: ProviderKind::GoogleGemini,
                credential: Credential::ApiKey {
                    header: None,
                    key: "test-key-2".to_string(),
                },
                default_model: Some("gemini-2.0-flash".to_string()),
                base_url: None,
                extra: HashMap::new(),
            },
        ];

        let client = build_client_from_configs(&configs, transport).expect("client");
        let mut handles = client.handles();
        handles.sort();

        assert_eq!(
            handles,
            vec!["gemini-default".to_string(), "openai-default".to_string()]
        );
    }

    #[test]
    fn build_client_from_configs_requires_credential() {
        let transport = default_dyn_transport().expect("transport");

        let configs = vec![ModelConfig {
            handle: "openai-no-cred".to_string(),
            provider: ProviderKind::OpenAiChat,
            credential: Credential::None,
            default_model: None,
            base_url: None,
            extra: HashMap::new(),
        }];

        let result = build_client_from_configs(&configs, transport);
        let error = match result {
            Ok(_) => panic!("expected auth error"),
            Err(err) => err,
        };
        match error {
            LLMError::Auth { message } => {
                assert!(
                    message.contains("openai_chat"),
                    "unexpected auth message: {message}"
                );
            }
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    /// 使用 ServiceAccount 凭证时应当被拒绝并返回 Auth 错误
    #[test]
    fn build_client_from_configs_rejects_service_account() {
        let transport = default_dyn_transport().expect("transport");

        let configs = vec![ModelConfig {
            handle: "gemini-service-account".to_string(),
            provider: ProviderKind::GoogleGemini,
            credential: Credential::ServiceAccount {
                json: serde_json::json!({ "type": "service_account" }),
            },
            default_model: Some("gemini-2.0-flash".to_string()),
            base_url: None,
            extra: HashMap::new(),
        }];

        let result = build_client_from_configs(&configs, transport);
        let error = match result {
            Ok(_) => panic!("expected auth error"),
            Err(err) => err,
        };

        match error {
            LLMError::Auth { message } => {
                assert!(
                    message.contains("google_gemini"),
                    "unexpected auth message for service account: {message}"
                );
            }
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    /// Bearer Token 凭证应当被接受并构建成功
    #[test]
    fn build_client_from_configs_accepts_bearer_token() {
        let transport = default_dyn_transport().expect("transport");

        let configs = vec![ModelConfig {
            handle: "openai-responses-bearer".to_string(),
            provider: ProviderKind::OpenAiResponses,
            credential: Credential::Bearer {
                token: "test-bearer-token".to_string(),
            },
            default_model: Some("gpt-4.1-mini".to_string()),
            base_url: None,
            extra: HashMap::new(),
        }];

        let result = build_client_from_configs(&configs, transport);
        if let Err(err) = result {
            panic!("expected ok for bearer token but got error: {err:?}");
        }
    }
}
