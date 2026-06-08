//! Database schema and migrations.

use rusqlite::Connection;

/// Run all pending schema migrations.
pub fn migrate(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "
        -- Memories table: stores all memory units
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            fact        TEXT,
            context     TEXT,
            metadata    TEXT NOT NULL DEFAULT '{}',
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
        -- Dimension 1024 matches DashScope text-embedding-v3 default.
        -- For OpenAI text-embedding-3-small, change to float[1536].
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            embedding float[1024]
        );
        ",
    )?;

    Ok(())
}
