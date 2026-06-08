//! Core data types for the Dentate memory system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single memory unit stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub fact: Option<String>,
    pub context: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub relevance_score: Option<f32>,
}

/// The result of a recall operation.
#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub memories: Vec<Memory>,
    pub total_found: usize,
    pub strategy_used: SearchStrategy,
}

/// The result of a reflect operation.
#[derive(Debug, Clone, Serialize)]
pub struct ReflectResult {
    pub answer: String,
    pub sources: Vec<Memory>,
    pub token_usage: TokenUsage,
}

/// Which retrieval strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum SearchStrategy {
    #[default]
    Hybrid,
    Semantic,
    Keyword,
}

use std::str::FromStr;

impl FromStr for SearchStrategy {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s.to_lowercase().as_str() {
            "semantic" | "vector" => Self::Semantic,
            "keyword" | "fts" | "text" => Self::Keyword,
            _ => Self::Hybrid,
        })
    }
}

/// Token usage for an LLM call.
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Configuration for the memory bank.
#[derive(Debug, Clone)]
pub struct BankConfig {
    pub database_path: String,
    pub llm_provider: String,
    pub llm_model: String,
    pub llm_base_url: Option<String>,
    /// Embedding provider: "openai", "dashscope", "zhipu", or "none".
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_base_url: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub enable_reranker: bool,
    /// Embedding vector dimension (1024 for DashScope, 1536 for OpenAI, etc.).
    pub vec_dim: usize,
}

impl BankConfig {
    pub fn from_config(cfg: &crate::config::Config) -> Self {
        let llm_key = crate::config::llm_api_key(cfg);
        let emb_key = crate::config::embeddings_api_key(cfg);

        if let Some(ref key) = llm_key {
            unsafe { std::env::set_var("DENTATE_LLM_API_KEY", key); }
        }
        if let Some(ref key) = emb_key {
            unsafe { std::env::set_var("DENTATE_EMBEDDINGS_API_KEY", key); }
        }

        let provider = cfg.embeddings.provider.clone().unwrap_or_else(|| "openai".into());
        let emb_base = cfg.embeddings.base_url.clone().or_else(|| match provider.as_str() {
            "dashscope" => Some("https://dashscope.aliyuncs.com/compatible-mode/v1".into()),
            "zhipu" => Some("https://open.bigmodel.cn/api/paas/v4".into()),
            _ => Some("https://api.openai.com/v1".into()),
        });

        // Dimension from config, or provider default
        let vec_dim = cfg.embeddings.dimensions.map(|d| d as usize).unwrap_or_else(|| {
            match provider.as_str() {
                "openai" => 1536,
                _ => 1024, // dashscope, zhipu, and others default to 1024
            }
        });

        Self {
            database_path: cfg.database.path.clone(),
            llm_provider: cfg.llm.provider.clone(),
            llm_model: cfg.llm.model.clone(),
            llm_base_url: cfg.llm.base_url.clone(),
            embedding_provider: provider,
            embedding_model: cfg.embeddings.model.clone(),
            embedding_base_url: emb_base,
            embedding_dimensions: cfg.embeddings.dimensions,
            enable_reranker: cfg.reranker.enabled,
            vec_dim,
        }
    }

    pub fn embeddings_disabled(&self) -> bool {
        self.embedding_provider == "none"
    }
}

impl Default for BankConfig {
    fn default() -> Self {
        let db = crate::config::default_db_path();
        Self {
            database_path: db,
            llm_provider: "deepseek".into(),
            llm_model: "deepseek-v4-flash".into(),
            llm_base_url: Some("https://api.deepseek.com/v1".into()),
            embedding_provider: "dashscope".into(),
            embedding_model: "text-embedding-v3".into(),
            embedding_base_url: Some("https://dashscope.aliyuncs.com/compatible-mode/v1".into()),
            embedding_dimensions: None,
            enable_reranker: false,
            vec_dim: 1024,
        }
    }
}
