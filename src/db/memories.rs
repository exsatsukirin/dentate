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
                id: row.get(0)?,
                content: row.get(1)?,
                fact: row.get(2)?,
                context: row.get(3)?,
                metadata: row
                    .get::<_, String>(4)
                    .and_then(|s| Ok(serde_json::from_str(&s).unwrap_or_default()))
                    .unwrap_or_default(),
                created_at: row
                    .get::<_, String>(5)
                    .and_then(|s| {
                        Ok(chrono::DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_default())
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
}
