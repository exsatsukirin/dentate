//! Retain: store new memories with optional fact extraction.

use crate::engine::MemoryBank;
use crate::types::Memory;

impl MemoryBank {
    /// Store a new memory.
    ///
    /// When embeddings are disabled, only FTS5 keyword search is available.
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

        // Generate embeddings only if configured
        let embeddings = if let Some(ref emb_client) = self.embeddings {
            Some(
                emb_client
                    .embed_batch(&facts.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                    .await?,
            )
        } else {
            None
        };

        let mut memories = Vec::new();
        let db = self.db.lock().await;

        for (i, fact) in facts.iter().enumerate() {
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

            crate::db::MemoryStore::insert(&db, &memory)?;

            // Insert embedding if available
            if let Some(ref emb_list) = embeddings {
                let emb_json = serde_json::to_string(&emb_list[i])?;
                db.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES (last_insert_rowid(), ?1)",
                    rusqlite::params![emb_json],
                )?;
            }

            memories.push(memory);
        }

        Ok(memories)
    }
}
