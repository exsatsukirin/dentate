//! Retain: store new memories with optional fact extraction.

use crate::engine::MemoryBank;
use crate::types::Memory;

impl MemoryBank {
    /// Store a new memory.
    ///
    /// * `content` - The content to remember.
    /// * `context` - Optional context (e.g. where this information came from).
    /// * `extract_facts` - If true, use the LLM to extract atomic facts from the content.
    ///
    /// Returns the stored memories (one per extracted fact, or one for the raw content).
    pub async fn retain(
        &self,
        content: &str,
        context: Option<&str>,
        extract_facts: bool,
    ) -> anyhow::Result<Vec<Memory>> {
        let facts: Vec<String> = if extract_facts {
            self.llm.extract_facts(content, context).await?
        } else {
            vec![content.to_string()]
        };

        let embeddings = self.embeddings.embed_batch(
            &facts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ).await?;

        let mut memories = Vec::new();
        let db = self.db.lock().await;

        for (fact, embedding) in facts.iter().zip(embeddings.iter()) {
            let id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now();

            let memory = Memory {
                id: id.clone(),
                content: content.to_string(),
                fact: Some(fact.clone()),
                context: context.map(|s| s.to_string()),
                metadata: serde_json::json!({}),
                created_at: now,
                relevance_score: None,
            };

            // Insert into memories table
            crate::db::MemoryStore::insert(&db, &memory)?;

            // Insert embedding into vec0 virtual table
            let emb_json = serde_json::to_string(embedding)?;
            db.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (last_insert_rowid(), ?1)",
                rusqlite::params![emb_json],
            )?;

            memories.push(memory);
        }

        Ok(memories)
    }
}
