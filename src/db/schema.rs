//! Database schema and migrations.

use rusqlite::Connection;

/// Run schema migrations with the given embedding vector dimension.
///
/// The `vec_dim` must match the embedding model in use (1024 for DashScope,
/// 1536 for OpenAI text-embedding-3-small, etc.). On first run the vec0 table
/// is created with this dimension. On subsequent runs, if a vec0 table already
/// exists, returns an error if the dimension doesn't match.
pub fn migrate(conn: &Connection, vec_dim: usize) -> anyhow::Result<()> {
    conn.execute_batch(&format!(
        "
        -- Memories table: stores all memory units
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            fact        TEXT,
            context     TEXT,
            metadata    TEXT NOT NULL DEFAULT '{{}}',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Full-text search index (FTS5) for keyword search
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            fact,
            context,
            content='memories',
            content_rowid='rowid'
        );

        -- Triggers to keep FTS5 in sync with memories table
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, fact, context)
            VALUES (new.rowid, new.content, new.fact, new.context);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, fact, context)
            VALUES ('delete', old.rowid, old.content, old.fact, old.context);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, fact, context)
            VALUES ('delete', old.rowid, old.content, old.fact, old.context);
            INSERT INTO memories_fts(rowid, content, fact, context)
            VALUES (new.rowid, new.content, new.fact, new.context);
        END;

        -- Vector index for semantic search (sqlite-vec vec0 virtual table).
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            embedding float[{vec_dim}]
        );

        -- Trigger to clean up vec_memories when memories are deleted
        CREATE TRIGGER IF NOT EXISTS vec_memories_ad AFTER DELETE ON memories BEGIN
            DELETE FROM vec_memories WHERE rowid = old.rowid;
        END;
        ",
    ))?;

    // Verify dimension consistency on re-open
    let existing_dim: Option<usize> = conn
        .query_row(
            "SELECT dimension FROM vec_memories_dims LIMIT 1",
            [],
            |row| row.get::<_, i64>(0).map(|d| d as usize),
        )
        .ok();

    if let Some(dim) = existing_dim
        && dim != vec_dim
    {
        anyhow::bail!(
            "vec_memories dimension mismatch: database has {dim}, config expects {vec_dim}. \
             Either update the embedding model to match, or delete the database and re-create it."
        );
    }

    Ok(())
}
