//! Recall: search memories by query.

use crate::db::SearchIndex;
use crate::engine::MemoryBank;
use crate::types::{Memory, RecallResult, SearchStrategy};

impl MemoryBank {
    /// Search for memories relevant to a query.
    ///
    /// When embeddings are disabled, all strategies fall back to keyword (FTS5) search.
    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        strategy: SearchStrategy,
    ) -> anyhow::Result<RecallResult> {
        let db = self.db.lock().await;

        // If embeddings are disabled, always use keyword search
        let effective_strategy = if self.embeddings.is_none() {
            SearchStrategy::Keyword
        } else {
            strategy
        };

        match effective_strategy {
            SearchStrategy::Hybrid => {
                let embedding = self
                    .embeddings
                    .as_ref()
                    .unwrap()
                    .embed(query)
                    .await?;

                let vec_results = SearchIndex::vector_search(&db, &embedding, limit * 2)?;
                let fts_results = SearchIndex::fts5_search(&db, query, limit * 2)?;

                let fused = SearchIndex::fuse(&[vec_results, fts_results], 60.0, limit);
                let memories: Vec<Memory> = fused.into_iter().map(|(m, _)| m).collect();
                let total = memories.len();

                Ok(RecallResult {
                    memories,
                    total_found: total,
                    strategy_used: SearchStrategy::Hybrid,
                })
            }
            SearchStrategy::Semantic => {
                let embedding = self
                    .embeddings
                    .as_ref()
                    .unwrap()
                    .embed(query)
                    .await?;
                let results = SearchIndex::vector_search(&db, &embedding, limit)?;
                let memories: Vec<Memory> = results.into_iter().map(|(m, _)| m).collect();
                let total = memories.len();

                Ok(RecallResult {
                    memories,
                    total_found: total,
                    strategy_used: SearchStrategy::Semantic,
                })
            }
            SearchStrategy::Keyword => {
                let results = SearchIndex::fts5_search(&db, query, limit)?;
                let memories: Vec<Memory> = results.into_iter().map(|(m, _)| m).collect();
                let total = memories.len();

                Ok(RecallResult {
                    memories,
                    total_found: total,
                    strategy_used: SearchStrategy::Keyword,
                })
            }
        }
    }
}
