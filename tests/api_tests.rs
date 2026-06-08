//! Integration tests for API clients.
//!
//! These tests require API keys set in environment variables.
//! If the required key is not set, tests are skipped.
//!
//! Required env vars:
//!   DEEPSEEK_API_KEY  or  OPENAI_API_KEY  — for LLM + Embeddings tests
//!   COHERE_API_KEY                          — for Reranker tests (optional)

use dentate::api::{EmbeddingsClient, LlmClient, RerankerClient};
use dentate::config;
use dentate::types::BankConfig;

/// Skip the test if no LLM API key is available.
fn require_api_key() -> Option<String> {
    let cfg = config::load_or_default();
    config::llm_api_key(&cfg)
}

/// Build an EmbeddingsClient from config file + env.
fn make_embeddings_client() -> Option<EmbeddingsClient> {
    let cfg = config::load_or_default();
    let bc = BankConfig::from_config(&cfg);
    if bc.embeddings_disabled() { return None; }
    let key = config::embeddings_api_key(&cfg)?;
    let base_url = bc.embedding_base_url.unwrap_or_else(|| "https://api.openai.com/v1".into());
    let model = bc.embedding_model;
    let dims = bc.embedding_dimensions;
    Some(EmbeddingsClient::new(&key, &base_url, &model, dims))
}

/// Skip the test if no Cohere API key is available.
fn require_cohere_key() -> Option<String> {
    let cfg = config::load_or_default();
    config::cohere_api_key(&cfg)
}

// ============================================================
// LLM tests
// ============================================================

#[tokio::test]
async fn test_extract_facts_basic() {
    let api_key = match require_api_key() {
        Some(k) => k,
        None => {
            eprintln!("SKIP: no API key set (set DEEPSEEK_API_KEY or OPENAI_API_KEY)");
            return;
        }
    };

    // Use model from config, fall back to deepseek-chat
    let cfg = config::load_or_default();
    let model = cfg.llm.model;
    let base_url = cfg.llm.base_url.unwrap_or_else(|| "https://api.deepseek.com/v1".into());
    let client = LlmClient::new(&api_key, &base_url, &model);

    let facts = client
        .extract_facts("Alice works at Google in Beijing. She is a senior software engineer.", None)
        .await
        .expect("fact extraction failed");

    assert!(!facts.is_empty(), "should extract at least one fact");
    // Each fact should be a non-empty string
    for fact in &facts {
        assert!(!fact.is_empty(), "fact should not be empty");
    }

    // At minimum, should extract "Alice works at Google" and the job title
    let joined = facts.join(" ").to_lowercase();
    assert!(joined.contains("alice"), "should mention Alice");
    assert!(joined.contains("google") || joined.contains("engineer"), "should mention work");
}

#[tokio::test]
async fn test_extract_facts_with_context() {
    let api_key = match require_api_key() {
        Some(k) => k,
        None => {
            eprintln!("SKIP: no API key set");
            return;
        }
    };

    // Use model from config, fall back to deepseek-chat
    let cfg = config::load_or_default();
    let model = cfg.llm.model;
    let base_url = cfg.llm.base_url.unwrap_or_else(|| "https://api.deepseek.com/v1".into());
    let client = LlmClient::new(&api_key, &base_url, &model);

    let facts = client
        .extract_facts("The deadline is next Friday.", Some("Project Alpha status meeting"))
        .await
        .expect("fact extraction failed");

    assert!(!facts.is_empty());
}

#[tokio::test]
async fn test_reflect_basic() {
    let api_key = match require_api_key() {
        Some(k) => k,
        None => {
            eprintln!("SKIP: no API key set");
            return;
        }
    };

    // Use model from config, fall back to deepseek-chat
    let cfg = config::load_or_default();
    let model = cfg.llm.model;
    let base_url = cfg.llm.base_url.unwrap_or_else(|| "https://api.deepseek.com/v1".into());
    let client = LlmClient::new(&api_key, &base_url, &model);

    let memories = vec![
        dentate::types::Memory {
            id: "m1".into(),
            content: "Alice works at Google".into(),
            fact: Some("Alice works at Google".into()),
            context: None,
            metadata: serde_json::json!({}),
            created_at: chrono::Utc::now(),
            relevance_score: None,
        },
        dentate::types::Memory {
            id: "m2".into(),
            content: "Alice is a software engineer".into(),
            fact: Some("Alice is a software engineer".into()),
            context: None,
            metadata: serde_json::json!({}),
            created_at: chrono::Utc::now(),
            relevance_score: None,
        },
    ];

    let (answer, usage) = client
        .reflect("What is Alice's job?", &memories)
        .await
        .expect("reflect failed");

    assert!(!answer.is_empty(), "should produce an answer");
    assert!(usage.input_tokens > 0, "should track input tokens");
    assert!(usage.output_tokens > 0, "should track output tokens");
}

// ============================================================
// Embeddings tests
// ============================================================

#[tokio::test]
async fn test_embed_single() {
    let client = match make_embeddings_client() {
        Some(c) => c,
        None => { eprintln!("SKIP: no API key set"); return; }
    };

    let embedding = client.embed("Hello world").await.expect("embedding failed");
    assert!(!embedding.is_empty(), "should return non-empty vector");
    // text-embedding-v3 default is 1024; text-embedding-3-small is 1536
    assert!(embedding.len() >= 256, "dim {} too small", embedding.len());
}

#[tokio::test]
async fn test_embed_batch() {
    let client = match make_embeddings_client() {
        Some(c) => c,
        None => { eprintln!("SKIP: no API key set"); return; }
    };

    let texts = ["Apple is a fruit", "Tesla is a car company", "Rust is a programming language"];
    let embeddings = client.embed_batch(&texts).await.expect("batch embedding failed");

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert!(!emb.is_empty());
        assert!(emb.len() >= 256, "dim {} too small", emb.len());
    }
}

#[tokio::test]
async fn test_embedding_semantic_similarity() {
    let client = match make_embeddings_client() {
        Some(c) => c,
        None => { eprintln!("SKIP: no API key set"); return; }
    };

    let apple = client.embed("Apple makes iPhones and MacBooks").await.unwrap();
    let orange = client.embed("Orange juice is a popular breakfast drink").await.unwrap();
    let car = client.embed("The Porsche 911 is a sports car").await.unwrap();

    let cos_sim = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };

    let apple_orange = cos_sim(&apple, &orange);
    let apple_car = cos_sim(&apple, &car);

    // Apple and orange are both consumer products — should be closer
    // than apple and car (different domains).
    // This is a weak assertion since small models are imperfect,
    // but it should hold for well-trained embedding models.
    assert!(
        apple_orange > apple_car * 0.8,
        "Apple-Orange similarity ({apple_orange:.3}) should not be drastically lower than Apple-Car ({apple_car:.3})"
    );
}

// ============================================================
// Reranker tests (Cohere)
// ============================================================

#[tokio::test]
async fn test_reranker_basic() {
    let api_key = match require_cohere_key() {
        Some(k) => k,
        None => {
            eprintln!("SKIP: no COHERE_API_KEY set");
            return;
        }
    };

    let client = RerankerClient::new(&api_key, "rerank-english-v3.0");

    let docs: Vec<String> = vec![
        "The Eiffel Tower is in Paris".into(),
        "Sushi is a Japanese dish".into(),
        "Python is a programming language".into(),
    ];

    let results = client
        .rerank("Tell me about France landmarks", &docs, 2)
        .await
        .expect("rerank failed");

    assert_eq!(results.len(), 2, "should return top_n results");
    assert_eq!(results[0].0, 0, "Eiffel Tower should rank first");

    // Relevance scores should be between 0 and 1
    for (_, score) in &results {
        assert!(*score > 0.0 && *score <= 1.0, "score {score} out of range");
    }
}
