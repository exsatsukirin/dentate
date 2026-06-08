//! Hybrid search — vector similarity + FTS5 full-text search.

use rusqlite::{params, Connection};

use crate::types::Memory;

/// Search index for vector + FTS5 queries.
pub struct SearchIndex;

impl SearchIndex {
    /// Full-text keyword search using FTS5.
    pub fn fts5_search(
        conn: &Connection,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(Memory, f32)>> {
        let safe_query = query.replace(['"', '*'], "");
        let fts_query = format!("\"{}\"", safe_query);

        let sql = "
            SELECT m.id, m.content, m.fact, m.context, m.metadata, m.created_at,
                   bm25(memories_fts) as score
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ?1
            ORDER BY score
            LIMIT ?2
        ";

        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
            let memory = Memory {
                id: row.get("id")?,
                content: row.get("content")?,
                fact: row.get("fact")?,
                context: row.get("context")?,
                metadata: row
                    .get::<_, String>("metadata")
                    .map(|s| serde_json::from_str(&s).unwrap_or_default())
                    .unwrap_or_default(),
                created_at: row
                    .get::<_, String>("created_at")
                    .map(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_default()
                    })
                    .unwrap_or_default(),
                relevance_score: Some(0.0),
            };
            let score: f64 = row.get("score")?;
            Ok((memory, score as f32))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Vector similarity search using sqlite-vec.
    pub fn vector_search(
        conn: &Connection,
        embedding: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<(Memory, f32)>> {
        let embedding_json = serde_json::to_string(embedding)?;

        let sql = "
            SELECT m.id, m.content, m.fact, m.context, m.metadata, m.created_at, v.distance
            FROM vec_memories v
            JOIN memories m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ?1 AND k = ?2
        ";

        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![embedding_json, limit as i64], |row| {
            let memory = Memory {
                id: row.get("id")?,
                content: row.get("content")?,
                fact: row.get("fact")?,
                context: row.get("context")?,
                metadata: row
                    .get::<_, String>("metadata")
                    .map(|s| serde_json::from_str(&s).unwrap_or_default())
                    .unwrap_or_default(),
                created_at: row
                    .get::<_, String>("created_at")
                    .map(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_default()
                    })
                    .unwrap_or_default(),
                relevance_score: Some(0.0),
            };
            let distance: f64 = row.get("distance")?;
            let relevance = 1.0 / (1.0 + distance as f32);
            Ok((memory, relevance))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Reciprocal Rank Fusion: merge results from multiple search strategies.
    pub fn fuse(
        results: &[Vec<(Memory, f32)>],
        k: f32,
        final_limit: usize,
    ) -> Vec<(Memory, f32)> {
        use std::collections::HashMap;

        let mut scores: HashMap<String, (Memory, f32)> = HashMap::new();

        for result_list in results {
            for (rank, (memory, _)) in result_list.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32);
                let entry = scores.entry(memory.id.clone()).or_insert_with(|| {
                    let mut m = memory.clone();
                    m.relevance_score = Some(0.0);
                    (m, 0.0)
                });
                entry.1 += rrf_score;
            }
        }

        let mut fused: Vec<(Memory, f32)> = scores.into_values().collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(final_limit);

        for (memory, score) in &mut fused {
            memory.relevance_score = Some(*score);
        }

        fused
    }
}
