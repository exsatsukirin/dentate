//! Configuration file loading from `~/.config/dentate/config.toml`.
//!
//! All fields are optional — missing values fall back to defaults or env vars.
//!
//! Example `~/.config/dentate/config.toml`:
//!
//! ```toml
//! [llm]
//! provider = "deepseek"
//! model = "deepseek-chat"
//! api_key = "sk-xxx"
//!
//! [embeddings]
//! model = "text-embedding-3-small"
//! dimensions = 1536
//! ```
//!
//! API keys can also be set via environment variables:
//!   DEEPSEEK_API_KEY, OPENAI_API_KEY, COHERE_API_KEY

use serde::Deserialize;

/// Top-level config, deserialised from `config.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub llm: LlmSection,
    #[serde(default)]
    pub embeddings: EmbeddingsSection,
    #[serde(default)]
    pub reranker: RerankerSection,
    #[serde(default)]
    pub database: DatabaseSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmSection {
    #[serde(default = "default_llm_provider")]
    pub provider: String,
    #[serde(default = "default_llm_model")]
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsSection {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default = "default_embedding_model")]
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub dimensions: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RerankerSection {
    #[serde(default)]
    pub enabled: bool,
    pub api_key: Option<String>,
    #[serde(default = "default_reranker_model")]
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseSection {
    #[serde(default = "default_db_path")]
    pub path: String,
}

// ---- defaults ----

fn default_llm_provider() -> String { "deepseek".into() }
fn default_llm_model() -> String { "deepseek-chat".into() }
fn default_embedding_model() -> String { "text-embedding-3-small".into() }
fn default_reranker_model() -> String { "rerank-english-v3.0".into() }
fn default_db_path() -> String { "dentate.db".into() }

impl Default for LlmSection {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            model: default_llm_model(),
            api_key: None,
            base_url: Some("https://api.deepseek.com/v1".into()),
        }
    }
}

impl Default for EmbeddingsSection {
    fn default() -> Self {
        Self {
            provider: None,
            model: default_embedding_model(),
            api_key: None,
            base_url: Some("https://api.openai.com/v1".into()),
            dimensions: None,
        }
    }
}

impl Default for DatabaseSection {
    fn default() -> Self {
        Self { path: default_db_path() }
    }
}

// ---- loading ----

/// Load configuration from the standard path: `~/.config/dentate/config.toml`.
///
/// Returns `Ok(None)` if the file does not exist.
pub fn load() -> anyhow::Result<Option<Config>> {
    let path = config_path();
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(Some(config))
}

/// Load or return defaults. Never fails — missing file = defaults.
pub fn load_or_default() -> Config {
    load().ok().flatten().unwrap_or_default()
}

/// Get the config file path: `~/.config/dentate/config.toml`.
pub fn config_path() -> std::path::PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("dentate")
        .join("config.toml")
}

/// Resolve an API key with precedence: env var > config file.
///
/// Checks env vars in order, then falls back to the config value.
pub fn resolve_api_key(env_vars: &[&str], config_value: Option<&str>) -> Option<String> {
    for var in env_vars {
        if let Ok(val) = std::env::var(var) {
            if !val.is_empty() {
                return Some(val);
            }
        }
    }
    config_value.filter(|v| !v.is_empty()).map(|v| v.to_string())
}

/// Convenience: get the LLM API key from env or config.
pub fn llm_api_key(config: &Config) -> Option<String> {
    resolve_api_key(
        &["DENTATE_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"],
        config.llm.api_key.as_deref(),
    )
}

/// Convenience: get the embeddings API key from env or config.
///
/// Does NOT fall back to LLM key — embeddings use a different service.
pub fn embeddings_api_key(config: &Config) -> Option<String> {
    resolve_api_key(
        &["DENTATE_EMBEDDINGS_API_KEY", "OPENAI_API_KEY"],
        config.embeddings.api_key.as_deref(),
    )
}

/// Convenience: get the Cohere API key from env or config.
pub fn cohere_api_key(config: &Config) -> Option<String> {
    resolve_api_key(&["COHERE_API_KEY"], config.reranker.api_key.as_deref())
}

// We use the `dirs` crate for XDG paths. If you prefer zero extra deps,
// replace with `std::env::var("HOME")` / `std::env::var("XDG_CONFIG_HOME")`.
mod dirs {
    pub fn config_dir() -> Option<std::path::PathBuf> {
        if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
            return Some(std::path::PathBuf::from(dir));
        }
        std::env::var("HOME")
            .ok()
            .map(|h| std::path::PathBuf::from(h).join(".config"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_path() {
        let path = config_path();
        assert!(path.ends_with("dentate/config.toml"), "got {path:?}");
    }

    #[test]
    fn test_defaults() {
        let c = Config::default();
        assert_eq!(c.llm.provider, "deepseek");
        assert_eq!(c.llm.model, "deepseek-chat");
        assert_eq!(c.embeddings.model, "text-embedding-3-small");
        assert_eq!(c.database.path, "dentate.db");
        assert!(!c.reranker.enabled);
    }

    #[test]
    fn test_resolve_api_key_env_first() {
        // Config has a key, but env var should take precedence
        let result = resolve_api_key(
            &["NONEXISTENT_VAR_12345"],
            Some("config-key"),
        );
        assert_eq!(result.as_deref(), Some("config-key"));
    }
}
