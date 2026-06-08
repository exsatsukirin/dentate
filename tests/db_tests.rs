//! Integration tests for the database layer.
//!
//! These tests run without any API keys — they verify SQLite schema,
//! CRUD operations, FTS5 full-text search, and sqlite-vec vector search.

use chrono::Utc;
use dentate::db::{MemoryStore, SearchIndex};

/// Create an in-memory test database.
fn test_db() -> rusqlite::Connection {
    dentate::db::open_in_memory(1024).expect("failed to open in-memory test database")
}

/// Create a test memory with a random ID.
fn test_memory(id: &str, content: &str, fact: &str) -> dentate::types::Memory {
    dentate::types::Memory {
        id: id.to_string(),
        content: content.to_string(),
        fact: Some(fact.to_string()),
        context: None,
        metadata: serde_json::json!({"test": true}),
        created_at: Utc::now(),
        relevance_score: None,
    }
}

// ============================================================
// Schema tests
// ============================================================

#[test]
fn test_schema_tables_exist() {
    let db = test_db();
    let tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    assert!(tables.contains(&"memories".to_string()), "memories table missing");
    assert!(
        tables.contains(&"memories_fts".to_string()),
        "FTS5 index missing"
    );
    // Note: vec0 virtual table is created via sqlite-vec auto-extension,
    // but its name depends on the vec0 configuration.
}

#[test]
fn test_vec_extension_loaded() {
    let db = test_db();
    let version: String = db
        .query_row("SELECT vec_version()", [], |row| row.get(0))
        .unwrap();
    assert!(version.starts_with('v'), "vec_version should start with 'v', got: {version}");
}

// ============================================================
// CRUD tests
// ============================================================

#[test]
fn test_insert_and_get_memory() {
    let db = test_db();
    let mem = test_memory("m1", "Alice works at Google", "Alice works at Google");
    MemoryStore::insert(&db, &mem).unwrap();
    assert_eq!(MemoryStore::count(&db).unwrap(), 1);

    let fetched = MemoryStore::get(&db, "m1").unwrap().expect("memory not found");
    assert_eq!(fetched.id, "m1");
    assert_eq!(fetched.content, "Alice works at Google");
    assert_eq!(fetched.fact.as_deref(), Some("Alice works at Google"));
}

#[test]
fn test_insert_multiple_and_count() {
    let db = test_db();
    for i in 0..5 {
        let mem = test_memory(&format!("m{i}"), &format!("content {i}"), &format!("fact {i}"));
        MemoryStore::insert(&db, &mem).unwrap();
    }
    assert_eq!(MemoryStore::count(&db).unwrap(), 5);
}

#[test]
fn test_get_nonexistent() {
    let db = test_db();
    assert!(MemoryStore::get(&db, "no-such-id").unwrap().is_none());
}

#[test]
fn test_delete_memory() {
    let db = test_db();
    let mem = test_memory("m1", "temp", "temp");
    MemoryStore::insert(&db, &mem).unwrap();
    assert_eq!(MemoryStore::count(&db).unwrap(), 1);
    MemoryStore::delete(&db, "m1").unwrap();
    assert_eq!(MemoryStore::count(&db).unwrap(), 0);
}

// ============================================================
// FTS5 full-text search tests
// ============================================================

#[test]
fn test_fts5_search_basic() {
    let db = test_db();
    MemoryStore::insert(&db, &test_memory("a", "Alice works at Google in Beijing", "Alice works at Google")).unwrap();
    MemoryStore::insert(&db, &test_memory("b", "Bob works at Apple in Shanghai", "Bob works at Apple")).unwrap();
    MemoryStore::insert(&db, &test_memory("c", "Charlie is a freelancer", "Charlie is a freelancer")).unwrap();

    let results = SearchIndex::fts5_search(&db, "Google", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "a");

    let results = SearchIndex::fts5_search(&db, "works", 10).unwrap();
    assert_eq!(results.len(), 2); // Alice and Bob
}

#[test]
fn test_fts5_search_no_match() {
    let db = test_db();
    MemoryStore::insert(&db, &test_memory("a", "Alice works at Google", "Alice works at Google")).unwrap();

    let results = SearchIndex::fts5_search(&db, "Microsoft", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_fts5_search_large_query() {
    let db = test_db();
    for i in 0..20 {
        MemoryStore::insert(
            &db,
            &test_memory(&format!("m{i}"), &format!("document number {i}"), &format!("doc {i}")),
        )
        .unwrap();
    }

    let results = SearchIndex::fts5_search(&db, "document", 5).unwrap();
    assert_eq!(results.len(), 5, "should respect limit");
}

// ============================================================
// Reciprocal Rank Fusion tests
// ============================================================

#[test]
fn test_rrf_fusion() {
    let mem1 = test_memory("a", "AAA", "AAA");
    let mem2 = test_memory("b", "BBB", "BBB");
    let mem3 = test_memory("c", "CCC", "CCC");

    let list1 = vec![(mem1.clone(), 0.9), (mem2.clone(), 0.5)];
    let list2 = vec![(mem2.clone(), 0.8), (mem3.clone(), 0.3)];

    let fused = SearchIndex::fuse(&[list1, list2], 60.0, 10);
    assert_eq!(fused.len(), 3);

    // mem2 appears in both lists — should rank highest
    assert_eq!(fused[0].0.id, "b", "mem2 should win RRF (appears in both lists)");
}

#[test]
fn test_rrf_fusion_single_list() {
    let mems: Vec<_> = (0..5)
        .map(|i| {
            let m = test_memory(&format!("m{i}"), &format!("doc {i}"), &format!("doc {i}"));
            (m, 1.0 - i as f32 * 0.1)
        })
        .collect();

    let fused = SearchIndex::fuse(&[mems], 60.0, 3);
    assert_eq!(fused.len(), 3);
    // Order should be preserved
    assert_eq!(fused[0].0.id, "m0");
    assert_eq!(fused[1].0.id, "m1");
    assert_eq!(fused[2].0.id, "m2");
}
