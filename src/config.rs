use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::LLMClient;
use crate::error::LLMError;
use crate::http::DynHttpTransport;

// Import the macro to register providers
use crate::register_providers;

// Register all providers using the macro
register_providers!(
    (openai_chat, "openai_chat", OpenAiChatProvider, OpenAiChat),
    (
        openai_responses,
        "openai_responses",
        OpenAiResponsesProvider,
        OpenAiResponses
    ),
    (
        anthropic_messages,
        "anthropic_messages",
        AnthropicMessagesProvider,
        AnthropicMessages
    ),
    (
        google_gemini,
        "google_gemini",
        GoogleGeminiProvider,
        GoogleGemini
    ),
);

/// Describes a provider handle that can be registered on an [`LLMClient`].
///
/// Each configuration declares the provider kind, credentials, optional defaults, and
/// vendor-specific metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Human-readable handle such as `default-openai`.
    pub handle: String,
    pub provider: ProviderKind,
    pub credential: Credential,
    pub default_model: Option<String>,
    pub base_url: Option<String>,
    /// Extra provider-specific settings such as `service_tier` or `safetySettings`.
    #[serde(default)]
    pub extra: HashMap<String, Value>,
    /// 运行时请求修改
    pub patch: Option<RequestPatch>,
}

/// Credential variants understood by the configuration loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    /// API key passed via an HTTP header.
    ApiKey {
        /// Optional header name; when omitted the provider default is used.
        header: Option<String>,
        /// Secret value used as the API key.
        key: String,
    },
    /// Bearer token used with the standard `Authorization` header.
    Bearer { token: String },
    /// Google or GCP Service Account JSON blob.
    ServiceAccount { json: Value },
    /// Dummy variant for providers that do not require credentials.
    None,
}

/// 声明式请求补丁，可在运行时重写 URL、Headers 或 Body。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestPatch {
    /// 覆盖请求 URL
    pub url: Option<String>,
    /// 深度合并到请求 body
    pub body: Option<Value>,
    /// None 表示删除 header，Some 则写入/覆盖
    pub headers: Option<HashMap<String, Option<String>>>,
    /// 按点分路径删除 body 中的字段（支持数组下标）
    pub remove_fields: Option<Vec<String>>,
}

impl RequestPatch {
    /// Applies the patch to the given request parts.
    pub fn apply(&self, url: &mut String, headers: &mut HashMap<String, String>, body: &mut Value) {
        if let Some(new_url) = &self.url {
            *url = new_url.clone();
        }

        if let Some(patch_body) = &self.body {
            merge_json(body, patch_body);
        }

        if let Some(header_patch) = &self.headers {
            for (key, value) in header_patch {
                match value {
                    Some(v) => {
                        headers.insert(key.clone(), v.clone());
                    }
                    None => {
                        headers.remove(key);
                    }
                }
            }
        }

        if let Some(paths) = &self.remove_fields {
            for path in paths {
                remove_json_field(body, path);
            }
        }
    }
}

fn merge_json(target: &mut Value, patch: &Value) {
    match patch {
        Value::Object(patch_map) => {
            if let Value::Object(target_map) = target {
                for (key, patch_value) in patch_map {
                    match target_map.get_mut(key) {
                        Some(existing) => merge_json(existing, patch_value),
                        None => {
                            target_map.insert(key.clone(), patch_value.clone());
                        }
                    }
                }
            } else {
                *target = patch.clone();
            }
        }
        _ => {
            *target = patch.clone();
        }
    }
}

fn remove_json_field(value: &mut Value, path: &str) {
    let segments: Vec<&str> = path
        .split('.')
        .filter(|segment| !segment.is_empty())
        .collect();
    if segments.is_empty() {
        return;
    }
    remove_json_field_inner(value, &segments);
}

fn remove_json_field_inner(value: &mut Value, segments: &[&str]) {
    if segments.is_empty() {
        return;
    }

    match value {
        Value::Object(map) => {
            if segments.len() == 1 {
                map.remove(segments[0]);
            } else if let Some(next) = map.get_mut(segments[0]) {
                remove_json_field_inner(next, &segments[1..]);
            }
        }
        Value::Array(array) => {
            if let Ok(index) = segments[0].parse::<usize>() {
                if index < array.len() {
                    if segments.len() == 1 {
                        array.remove(index);
                    } else {
                        remove_json_field_inner(&mut array[index], &segments[1..]);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Builds an [`LLMClient`] from the provided model configurations.
///
/// Each configuration is converted into a `DynProvider` with the supplied transport and
/// registered under the declared handle, returning an [`LLMClient`] that owns all
/// resulting providers. The helper is ideal for bootstrapping clients from configuration
/// files or environment settings.
///
/// # Examples
///
/// ```
/// # use std::collections::HashMap;
/// # use kotoba_llm::config::{ModelConfig, ProviderKind, Credential, build_client_from_configs};
/// # use kotoba_llm::http::reqwest::default_dyn_transport;
/// let configs = vec![ModelConfig {
///     handle: "default-openai".into(),
///     provider: ProviderKind::OpenAiChat,
///     credential: Credential::ApiKey { header: None, key: "test-key".into() },
///     default_model: Some("gpt-4.1-mini".into()),
///     base_url: None,
///     extra: HashMap::new(),
///     patch: None,
/// }];
/// let transport = default_dyn_transport().expect("transport");
/// let client = build_client_from_configs(&configs, transport).expect("client");
/// assert_eq!(client.handles(), vec!["default-openai".to_string()]);
/// ```
///
/// # Errors
///
/// Returns [`LLMError::Auth`] when credentials are invalid or missing, [`LLMError::InvalidConfig`]
/// when duplicate handles are present, or any provider-specific error raised while
/// constructing provider instances.
pub fn build_client_from_configs(
    configs: &[ModelConfig],
    transport: DynHttpTransport,
) -> Result<LLMClient, LLMError> {
    let mut builder = LLMClient::builder();

    for config in configs {
        let provider = build_provider_from_config(config, transport.clone())?;
        builder = builder.register_handle(config.handle.clone(), provider)?;
    }

    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::reqwest::default_dyn_transport;
    use serde_json::json;

    /// Ensures every [`ProviderKind`] variant can be registered on [`LLMClient`].
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
                patch: None,
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
                patch: None,
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
                patch: None,
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
                patch: None,
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
                patch: None,
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
                patch: None,
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
            patch: None,
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

    /// Rejects unsupported service account credentials.
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
            patch: None,
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

    /// Accepts bearer-token credentials when building the client.
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
            patch: None,
        }];

        let result = build_client_from_configs(&configs, transport);
        if let Err(err) = result {
            panic!("expected ok for bearer token but got error: {err:?}");
        }
    }

    /// Surfaces a validation error when duplicate handles exist in the input configs.
    #[test]
    fn build_client_from_configs_rejects_duplicate_handles() {
        let transport = default_dyn_transport().expect("transport");

        let cfg1 = ModelConfig {
            handle: "dup-handle".to_string(),
            provider: ProviderKind::OpenAiChat,
            credential: Credential::ApiKey {
                header: None,
                key: "key-1".to_string(),
            },
            default_model: Some("gpt-4.1-mini".to_string()),
            base_url: None,
            extra: HashMap::new(),
            patch: None,
        };

        let cfg2 = ModelConfig {
            handle: "dup-handle".to_string(),
            provider: ProviderKind::GoogleGemini,
            credential: Credential::ApiKey {
                header: None,
                key: "key-2".to_string(),
            },
            default_model: Some("gemini-2.0-flash".to_string()),
            base_url: None,
            extra: HashMap::new(),
            patch: None,
        };

        let configs = vec![cfg1, cfg2];
        let result = build_client_from_configs(&configs, transport);
        let err = match result {
            Ok(_) => panic!("expected duplicate handle error"),
            Err(err) => err,
        };

        match err {
            LLMError::InvalidConfig { field, reason } => {
                assert_eq!(field, "handle");
                assert!(
                    reason.contains("duplicate model handle: dup-handle"),
                    "unexpected invalid-config reason for duplicate handle: {reason}"
                );
            }
            other => panic!("unexpected error type for duplicate handle in configs: {other:?}"),
        }
    }

    #[test]
    fn request_patch_merges_and_removes_fields() {
        let patch = RequestPatch {
            url: Some("https://override.local/chat".to_string()),
            body: Some(json!({
                "metadata": { "trace": false },
                "extra": { "temperature": 0.2 }
            })),
            headers: Some(HashMap::from([
                ("X-Debug".to_string(), None),
                ("X-Injected".to_string(), Some("1".to_string())),
            ])),
            remove_fields: Some(vec![
                "metadata.remove_me".to_string(),
                "messages.0.drop".to_string(),
            ]),
        };

        let mut url = "https://api.example.com/v1/chat".to_string();
        let mut headers = HashMap::from([
            ("Authorization".to_string(), "Bearer test".to_string()),
            ("X-Debug".to_string(), "true".to_string()),
        ]);
        let mut body = json!({
            "metadata": { "trace": true, "remove_me": "temp" },
            "messages": [{ "role": "user", "drop": "yes" }],
        });

        patch.apply(&mut url, &mut headers, &mut body);

        assert_eq!(url, "https://override.local/chat");
        assert!(!headers.contains_key("X-Debug"));
        assert_eq!(headers.get("X-Injected"), Some(&"1".to_string()));
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer test".to_string())
        );
        assert_eq!(body["metadata"]["trace"], json!(false));
        assert!(body["metadata"].get("remove_me").is_none());
        assert!(body["messages"][0].get("drop").is_none());
        assert_eq!(body["extra"]["temperature"], json!(0.2));
    }

    #[test]
    fn request_patch_replaces_scalar_body() {
        let patch = RequestPatch {
            url: None,
            body: Some(json!({ "structured": true })),
            headers: None,
            remove_fields: None,
        };

        let mut url = "https://example.com".to_string();
        let mut headers = HashMap::new();
        let mut body = json!("raw text body");

        patch.apply(&mut url, &mut headers, &mut body);
        assert_eq!(body, json!({ "structured": true }));
        assert!(headers.is_empty());
        assert_eq!(url, "https://example.com");
    }
}
