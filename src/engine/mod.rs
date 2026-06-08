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
///
/// Manages the full lifecycle: retain → recall → reflect.
/// Thread-safe, designed to be shared across async tasks.
pub struct MemoryBank {
    db: Arc<Mutex<Connection>>,
    llm: LlmClient,
    embeddings: EmbeddingsClient,
    #[allow(dead_code)]
    reranker: Option<RerankerClient>,
    config: BankConfig,
}

impl MemoryBank {
    /// Open (or create) a memory bank at the given database path.
    ///
    /// ```rust,no_run
    /// use dentate::MemoryBank;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let bank = MemoryBank::open("my_agent.db")?;
    ///     // ... use bank ...
    ///     Ok(())
    /// }
    /// ```
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let config = BankConfig {
            database_path: path.to_string(),
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Open a memory bank with full configuration control.
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

    /// Get the bank configuration.
    pub fn config(&self) -> &BankConfig {
        &self.config
    }

    /// Get a reference to the underlying database connection.
    pub fn db(&self) -> &Arc<Mutex<Connection>> {
        &self.db
    }
}
