//! End-to-end integration tests for the Dentate memory engine.
//!
//! These test the full retain → recall → reflect pipeline.
//! Requires API keys set in environment variables (skipped if missing).

use std::env::temp_dir;
use std::sync::atomic::{AtomicU32, Ordering};

use dentate::config;
use dentate::engine::MemoryBank;
use dentate::types::{BankConfig, SearchStrategy};

/// Generate a unique temp database path for each test.
fn temp_db_path() -> String {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    temp_dir()
        .join(format!("dentate_test_{pid}_{n}.db"))
        .to_str()
        .unwrap()
        .to_string()
}

fn make_config() -> BankConfig {
    let cfg = config::load_or_default();
    let mut bc = BankConfig::from_config(&cfg);
    bc.database_path = temp_db_path();
    bc
}

fn require_both_keys() -> Option<(String, String)> {
    let cfg = config::load_or_default();
    let llm = config::llm_api_key(&cfg)?;
    let emb = config::embeddings_api_key(&cfg)?;
    Some((llm, emb))
}

// ============================================================
// In-memory database tests (no API needed)
// ============================================================

#[test]
fn test_bank_open_and_config() {
    let config = make_config();
    // Just verify the config loaded from file — model name varies by user config
    assert!(!config.llm_model.is_empty());
    assert!(!config.embedding_model.is_empty());
}

#[test]
fn test_search_strategy_parsing() {
    let s: SearchStrategy = "hybrid".parse().unwrap(); assert_eq!(s, SearchStrategy::Hybrid);
    let s: SearchStrategy = "semantic".parse().unwrap(); assert_eq!(s, SearchStrategy::Semantic);
    let s: SearchStrategy = "vector".parse().unwrap(); assert_eq!(s, SearchStrategy::Semantic);
    let s: SearchStrategy = "keyword".parse().unwrap(); assert_eq!(s, SearchStrategy::Keyword);
    let s: SearchStrategy = "fts".parse().unwrap(); assert_eq!(s, SearchStrategy::Keyword);
    let s: SearchStrategy = "text".parse().unwrap(); assert_eq!(s, SearchStrategy::Keyword);
    // Unknown defaults to Hybrid
    let s: SearchStrategy = "unknown".parse().unwrap(); assert_eq!(s, SearchStrategy::Hybrid);
    let s: SearchStrategy = "".parse().unwrap(); assert_eq!(s, SearchStrategy::Hybrid);
    assert_eq!(SearchStrategy::default(), SearchStrategy::Hybrid);
}

// ============================================================
// Full pipeline tests (requires API key)
// ============================================================

#[tokio::test]
async fn test_retain_and_recall_hybrid() {
    let _keys = require_both_keys();
    if _keys.is_none() {
        eprintln!("SKIP: no API key set");
        return;
    }

    let bank = MemoryBank::with_config(make_config()).expect("failed to open bank");

    // Retain some memories
    bank.retain("Alice works at Google in Beijing as a software engineer", None, true)
        .await
        .expect("retain failed");

    bank.retain("Bob is a product manager at Apple in Shanghai", None, true)
        .await
        .expect("retain failed");

    bank.retain("Charlie is a freelance designer based in Chengdu", None, true)
        .await
        .expect("retain failed");

    // Recall: find Alice
    let results = bank
        .recall("Who works at Google?", 3, SearchStrategy::Hybrid)
        .await
        .expect("recall failed");

    assert!(!results.memories.is_empty(), "should find at least one memory");
    assert_eq!(results.strategy_used, SearchStrategy::Hybrid);

    // Alice should be in the top results
    let top_content = results.memories.iter()
        .flat_map(|m| m.fact.as_deref().or(Some(&m.content)))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    assert!(top_content.contains("alice") || top_content.contains("google"),
        "top results should mention Alice or Google, got: {top_content}");

    // Recall: keyword search
    let results = bank
        .recall("Beijing", 3, SearchStrategy::Keyword)
        .await
        .expect("keyword recall failed");

    // At minimum should return some results
    assert!(!results.memories.is_empty());
}

#[tokio::test]
async fn test_reflect_pipeline() {
    let _keys = require_both_keys();
    if _keys.is_none() {
        eprintln!("SKIP: no API key set");
        return;
    }

    let db_path = temp_db_path();

    let mut config = make_config();
    config.database_path = db_path;
    let bank = MemoryBank::with_config(config).expect("failed to open bank");

    // Store knowledge about Alice
    bank.retain("Alice is a software engineer who works remotely from Beijing", None, true)
        .await
        .expect("retain failed");
    bank.retain("Alice's favorite programming language is Rust", None, true)
        .await
        .expect("retain failed");

    // Reflect
    let result = bank
        .reflect("What do we know about Alice?")
        .await
        .expect("reflect failed");

    assert!(!result.answer.is_empty(), "should produce an answer");
    assert!(!result.sources.is_empty(), "should cite sources");
    assert!(result.token_usage.input_tokens > 0);
    assert!(result.token_usage.output_tokens > 0);

    // Answer should mention Alice
    let answer_lower = result.answer.to_lowercase();
    assert!(answer_lower.contains("alice"), "answer should mention Alice: {answer}", answer = result.answer);
}

#[tokio::test]
async fn test_retain_without_fact_extraction() {
    let _keys = require_both_keys();
    if _keys.is_none() {
        eprintln!("SKIP: no API key set");
        return;
    }

    let mut config = make_config();
    config.database_path = temp_db_path();
    let bank = MemoryBank::with_config(config).expect("failed to open bank");

    // Retain without fact extraction
    let memories = bank
        .retain("Raw content without fact extraction", None, false)
        .await
        .expect("retain failed");

    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].fact.as_deref(), Some("Raw content without fact extraction"));
    assert_eq!(memories[0].content, "Raw content without fact extraction");
}

// ============================================================
// No-embedding mode tests (only requires LLM key)
// ============================================================

#[tokio::test]
async fn test_retain_recall_no_embeddings() {
    let cfg = config::load_or_default();
    let llm_key = config::llm_api_key(&cfg);
    if llm_key.is_none() {
        eprintln!("SKIP: no API key set");
        return;
    }

    let bank = MemoryBank::with_config(BankConfig {
        embedding_provider: "none".into(),
        ..make_config()
    })
    .expect("failed to open bank");

    assert!(!bank.has_embeddings());

    // Retain
    bank.retain("Alice works at Google in Beijing", None, true)
        .await
        .expect("retain failed");
    bank.retain("Bob works at Apple in Shanghai", None, true)
        .await
        .expect("retain failed");

    // Recall with keyword (only strategy available)
    let results = bank
        .recall("Google", 5, SearchStrategy::Keyword)
        .await
        .expect("recall failed");

    assert!(!results.memories.is_empty());
    assert_eq!(results.strategy_used, SearchStrategy::Keyword);

    // Hybrid falls back to keyword automatically
    let results = bank
        .recall("Apple Shanghai", 5, SearchStrategy::Hybrid)
        .await
        .expect("recall failed");
    assert_eq!(results.strategy_used, SearchStrategy::Keyword);
}

#[test]
fn test_embeddings_disabled_flag() {
    let mut cfg = BankConfig::default();
    assert!(!cfg.embeddings_disabled());

    cfg.embedding_provider = "none".into();
    assert!(cfg.embeddings_disabled());
}
