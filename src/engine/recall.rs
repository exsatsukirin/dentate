//! Recall: search memories by query.

use crate::db::SearchIndex;
use crate::engine::MemoryBank;
use crate::types::{Memory, RecallResult, SearchStrategy};

impl MemoryBank {
    /// Search for memories relevant to a query.
    ///
    /// * `query` - The search query.
    /// * `limit` - Maximum number of results to return.
    /// * `strategy` - Which search strategy to use (Hybrid, Semantic, or Keyword).
    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        strategy: SearchStrategy,
    ) -> anyhow::Result<RecallResult> {
        let db = self.db.lock().await;

        match strategy {
            SearchStrategy::Hybrid => {
                // Run vector and keyword search in parallel-like fashion
                let embedding = self.embeddings.embed(query).await?;

                let vec_results = SearchIndex::vector_search(&db, &embedding, limit * 2)?;
                let fts_results = SearchIndex::fts5_search(&db, query, limit * 2)?;

                // Fuse results with Reciprocal Rank Fusion
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
                let embedding = self.embeddings.embed(query).await?;
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
