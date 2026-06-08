//! Core data types for the Dentate memory system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single memory unit stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier.
    pub id: String,
    /// The original content.
    pub content: String,
    /// LLM-extracted fact (if fact extraction was enabled during retain).
    pub fact: Option<String>,
    /// Optional context provided at retain time.
    pub context: Option<String>,
    /// Arbitrary JSON metadata.
    pub metadata: serde_json::Value,
    /// When this memory was created.
    pub created_at: DateTime<Utc>,
    /// Relevance score set during recall (None outside search results).
    pub relevance_score: Option<f32>,
}

/// The result of a recall operation.
#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    /// Matching memories, ordered by relevance (best first).
    pub memories: Vec<Memory>,
    /// Total number of results found before truncation.
    pub total_found: usize,
    /// The search strategy that was used.
    pub strategy_used: SearchStrategy,
}

/// The result of a reflect operation (generate an answer from memories).
#[derive(Debug, Clone, Serialize)]
pub struct ReflectResult {
    /// The generated answer.
    pub answer: String,
    /// Source memories that informed the answer.
    pub sources: Vec<Memory>,
    /// Token usage for the LLM call.
    pub token_usage: TokenUsage,
}

/// Which retrieval strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum SearchStrategy {
    /// Combine vector similarity + full-text search (recommended default).
    #[default]
    Hybrid,
    /// Pure vector (semantic) search.
    Semantic,
    /// Pure keyword (FTS5) search.
    Keyword,
}

impl SearchStrategy {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "semantic" | "vector" => Self::Semantic,
            "keyword" | "fts" | "text" => Self::Keyword,
            _ => Self::Hybrid,
        }
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
    /// Path to the SQLite database file.
    pub database_path: String,
    /// LLM provider: "openai", "deepseek", or any OpenAI-compatible.
    pub llm_provider: String,
    /// LLM model name (e.g. "gpt-4o-mini", "deepseek-chat").
    pub llm_model: String,
    /// LLM API base URL (defaults based on provider).
    pub llm_base_url: Option<String>,
    /// Embedding model (default: "text-embedding-3-small").
    pub embedding_model: String,
    /// Embedding dimensions (default: 1536, can reduce to 256 for cost savings).
    pub embedding_dimensions: Option<u32>,
    /// Whether to enable Cohere Rerank (default: false, uses vector scores).
    pub enable_reranker: bool,
}

impl BankConfig {
    /// Create a BankConfig from the config file, merging with env var overrides.
    pub fn from_config(cfg: &crate::config::Config) -> Self {
        let llm_key = crate::config::llm_api_key(cfg);
        let emb_key = crate::config::embeddings_api_key(cfg);

        // Set key as env var so API clients pick it up (avoids threading keys through)
        if let Some(ref key) = llm_key {
            unsafe { std::env::set_var("DENTATE_LLM_API_KEY", key); }
        }
        if let Some(ref key) = emb_key {
            unsafe { std::env::set_var("DENTATE_EMBEDDINGS_API_KEY", key); }
        }

        Self {
            database_path: cfg.database.path.clone(),
            llm_provider: cfg.llm.provider.clone(),
            llm_model: cfg.llm.model.clone(),
            llm_base_url: cfg.llm.base_url.clone(),
            embedding_model: cfg.embeddings.model.clone(),
            embedding_dimensions: cfg.embeddings.dimensions,
            enable_reranker: cfg.reranker.enabled,
        }
    }
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            database_path: "dentate.db".into(),
            llm_provider: "deepseek".into(),
            llm_model: "deepseek-chat".into(),
            llm_base_url: Some("https://api.deepseek.com/v1".into()),
            embedding_model: "text-embedding-3-small".into(),
            embedding_dimensions: None,
            enable_reranker: false,
        }
    }
}
