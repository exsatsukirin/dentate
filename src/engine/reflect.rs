//! Reflect: generate an answer from retrieved memories.

use crate::engine::MemoryBank;
use crate::types::ReflectResult;

impl MemoryBank {
    /// Reflect on a query using the agent's memory bank.
    ///
    /// This performs a hybrid recall internally, then asks the LLM to
    /// synthesize an answer from the retrieved memories.
    pub async fn reflect(&self, query: &str) -> anyhow::Result<ReflectResult> {
        // First, recall relevant memories
        let recall = self
            .recall(query, 10, crate::types::SearchStrategy::Hybrid)
            .await?;

        // Then, ask the LLM to reflect on them
        let (answer, token_usage) = self.llm.reflect(query, &recall.memories).await?;

        Ok(ReflectResult {
            answer,
            sources: recall.memories,
            token_usage,
        })
    }
}
