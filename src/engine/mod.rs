//! Memory engine — the core retain / recall / reflect pipeline.

mod retain;
mod recall;
mod reflect;

use std::sync::Arc;

use rusqlite::Connection;
use tokio::sync::Mutex;

use crate::api::{EmbeddingsClient, LlmClient, RerankerClient};
use crate::types::*;

/// The main memory bank — your agent's dentate gyrus.
pub struct MemoryBank {
    db: Arc<Mutex<Connection>>,
    llm: LlmClient,
    /// None when embeddings are disabled (provider = "none").
    embeddings: Option<EmbeddingsClient>,
    #[allow(dead_code)]
    reranker: Option<RerankerClient>,
    config: BankConfig,
}

impl MemoryBank {
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let config = BankConfig {
            database_path: path.to_string(),
            ..Default::default()
        };
        Self::with_config(config)
    }

    pub fn with_config(config: BankConfig) -> anyhow::Result<Self> {
        let conn = crate::db::open(&config.database_path)?;
        let llm = crate::api::create_llm_client(&config)?;
        let embeddings = crate::api::create_embeddings_client(&config)?;
        let reranker = if config.enable_reranker {
            let api_key = std::env::var("COHERE_API_KEY").unwrap_or_default();
            Some(RerankerClient::new(&api_key, "rerank-english-v3.0"))
        } else {
            None
        };

        Ok(Self {
            db: Arc::new(Mutex::new(conn)),
            llm,
            embeddings,
            reranker,
            config,
        })
    }

    pub fn config(&self) -> &BankConfig {
        &self.config
    }

    pub fn db(&self) -> &Arc<Mutex<Connection>> {
        &self.db
    }

    /// Returns true if embeddings are available.
    pub fn has_embeddings(&self) -> bool {
        self.embeddings.is_some()
    }
}
