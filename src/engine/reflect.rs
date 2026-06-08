//! Reflect: generate an answer from retrieved memories.

use crate::engine::MemoryBank;
use crate::types::ReflectResult;

impl MemoryBank {
    /// Reflect on a query using the agent's memory bank.
    pub async fn reflect(&self, query: &str) -> anyhow::Result<ReflectResult> {
        self.reflect_with_strategy(query, crate::types::SearchStrategy::Hybrid).await
    }

    /// Reflect with a specific retrieval strategy.
    pub async fn reflect_with_strategy(
        &self,
        query: &str,
        strategy: crate::types::SearchStrategy,
    ) -> anyhow::Result<ReflectResult> {
        let recall = self.recall(query, 10, strategy).await?;
        let (answer, token_usage) = self.llm.reflect(query, &recall.memories).await?;
        Ok(ReflectResult { answer, sources: recall.memories, token_usage })
    }
}
