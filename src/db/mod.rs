//! Database module — SQLite schema, migrations, and connection management.

use rusqlite::Connection;
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

mod memories;
mod search;
mod schema;

pub use memories::MemoryStore;
pub use search::SearchIndex;

/// Open (or create) the SQLite database and run migrations.
///
/// Automatically loads the sqlite-vec extension for vector search.
pub fn open(path: &str) -> anyhow::Result<Connection> {
    // Register sqlite-vec as an auto-extension (loaded for every new connection)
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }

    let conn = Connection::open(path)?;

    // Enable WAL mode for better concurrent read performance
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

    // Run schema migrations
    schema::migrate(&conn)?;

    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_in_memory() {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
        schema::migrate(&conn).unwrap();

        // Verify vec0 works
        let version: String = conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(version.starts_with('v'));
    }
}
