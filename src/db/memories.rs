//! Memory CRUD operations on the SQLite database.

use rusqlite::{params, Connection};

use crate::types::Memory;

/// Store and retrieve memory units.
pub struct MemoryStore;

impl MemoryStore {
    /// Insert a new memory. Returns the number of rows inserted.
    pub fn insert(conn: &Connection, memory: &Memory) -> anyhow::Result<usize> {
        let n = conn.execute(
            "INSERT INTO memories (id, content, fact, context, metadata, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                memory.id,
                memory.content,
                memory.fact,
                memory.context,
                memory.metadata.to_string(),
                memory.created_at.to_rfc3339(),
            ],
        )?;
        Ok(n)
    }

    /// Fetch a memory by ID.
    pub fn get(conn: &Connection, id: &str) -> anyhow::Result<Option<Memory>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, fact, context, metadata, created_at
             FROM memories WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Memory {
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
                relevance_score: None,
            })
        })?;
        match rows.next() {
            Some(Ok(memory)) => Ok(Some(memory)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Delete a memory by ID. Returns the number of rows deleted.
    pub fn delete(conn: &Connection, id: &str) -> anyhow::Result<usize> {
        let n = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(n)
    }

    /// Count total memories.
    pub fn count(conn: &Connection) -> anyhow::Result<usize> {
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        Ok(n as usize)
    }

    /// List memories with optional search query and pagination.
    pub fn list(
        conn: &Connection,
        query: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<Vec<Memory>> {
        let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(q) = query {
            let fts = format!("\"{}\"", q.replace('"', ""));
            (
                "SELECT m.id, m.content, m.fact, m.context, m.metadata, m.created_at
                 FROM memories_fts f JOIN memories m ON m.rowid = f.rowid
                 WHERE memories_fts MATCH ?1
                 ORDER BY rank LIMIT ?2 OFFSET ?3"
                    .into(),
                vec![Box::new(fts), Box::new(limit as i64), Box::new(offset as i64)],
            )
        } else {
            (
                "SELECT id, content, fact, context, metadata, created_at
                 FROM memories ORDER BY created_at DESC LIMIT ?1 OFFSET ?2"
                    .into(),
                vec![Box::new(limit as i64), Box::new(offset as i64)],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
            |row| {
                Ok(Memory {
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
                    relevance_score: None,
                })
            },
        )?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
