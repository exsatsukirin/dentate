//! API clients for external services (LLM, Embeddings, Reranker).

pub mod embeddings;
pub mod llm;
pub mod reranker;

pub use embeddings::EmbeddingsClient;
pub use llm::LlmClient;
pub use reranker::RerankerClient;

use crate::types::BankConfig;

/// Create an LLM client from configuration.
pub fn create_llm_client(config: &BankConfig) -> anyhow::Result<LlmClient> {
    let api_key = std::env::var("DENTATE_LLM_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .unwrap_or_default();

    let base_url = config
        .llm_base_url
        .clone()
        .unwrap_or_else(|| match config.llm_provider.as_str() {
            "deepseek" => "https://api.deepseek.com/v1".into(),
            "openai" => "https://api.openai.com/v1".into(),
            other => format!("https://api.{}.com/v1", other),
        });

    Ok(LlmClient::new(&api_key, &base_url, &config.llm_model))
}

/// Create an embeddings client from configuration.
pub fn create_embeddings_client(config: &BankConfig) -> anyhow::Result<EmbeddingsClient> {
    let api_key = std::env::var("DENTATE_EMBEDDINGS_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_default();

    // Embeddings always use OpenAI (or compatible) endpoint
    let base_url = std::env::var("DENTATE_EMBEDDINGS_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    Ok(EmbeddingsClient::new(
        &api_key,
        &base_url,
        &config.embedding_model,
        config.embedding_dimensions,
    ))
}
