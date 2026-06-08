//! Database schema and migrations.

use rusqlite::Connection;

/// Run schema migrations with the given embedding vector dimension.
pub fn migrate(conn: &Connection, vec_dim: usize) -> anyhow::Result<()> {
    conn.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            fact        TEXT,
            context     TEXT,
            metadata    TEXT NOT NULL DEFAULT '{{}}',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content, fact, context,
            content='memories', content_rowid='rowid'
        );

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

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            embedding float[{vec_dim}]
        );

        CREATE TRIGGER IF NOT EXISTS vec_memories_ad AFTER DELETE ON memories BEGIN
            DELETE FROM vec_memories WHERE rowid = old.rowid;
        END;

        -- Own metadata table (don't rely on sqlite-vec internals)
        CREATE TABLE IF NOT EXISTS dentate_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR REPLACE INTO dentate_meta (key, value)
        VALUES ('vec_dim', '{vec_dim}');
        ",
    ))?;

    // Verify dimension consistency on re-open
    let stored_dim: Option<String> = conn
        .query_row(
            "SELECT value FROM dentate_meta WHERE key = 'vec_dim'",
            [],
            |row| row.get(0),
        )
        .ok();

    if let Some(dim_str) = stored_dim {
        let dim: usize = dim_str.parse().unwrap_or(0);
        if dim != vec_dim && dim != 0 {
            anyhow::bail!(
                "vec_memories dimension mismatch: database has {dim}, config expects {vec_dim}. \
                 Either update the embedding model to match, or delete the database and re-create it."
            );
        }
    }

    Ok(())
}
