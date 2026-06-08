//! API clients for external services (LLM, Embeddings, Reranker).

pub mod embeddings;
pub mod llm;
pub mod reranker;

pub use embeddings::EmbeddingsClient;
pub use llm::LlmClient;
pub use reranker::RerankerClient;

use crate::types::BankConfig;

/// Retry a fallible async operation with exponential backoff.
///
/// Retries on transient failures (HTTP 429, 5xx) or network errors.
pub async fn retry_with_backoff<F, Fut, T>(mut f: F, max_attempts: u32) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut attempt = 0;
    loop {
        attempt += 1;
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) if attempt < max_attempts && is_retryable(&e) => {
                let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tracing::warn!("Retry {attempt}/{max_attempts} after {delay:?}: {e}");
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

fn is_retryable(e: &anyhow::Error) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("429")
        || msg.contains("503")
        || msg.contains("502")
        || msg.contains("rate limit")
        || msg.contains("timeout")
        || msg.contains("connection")
        || msg.contains("tls")
            && !msg.contains("401")
            && !msg.contains("403")
}

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
///
/// Returns `Ok(None)` when `embedding_provider` is `"none"`.
pub fn create_embeddings_client(
    config: &BankConfig,
) -> anyhow::Result<Option<EmbeddingsClient>> {
    if config.embeddings_disabled() {
        return Ok(None);
    }

    let api_key = std::env::var("DENTATE_EMBEDDINGS_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_default();

    let base_url = config
        .embedding_base_url
        .clone()
        .unwrap_or_else(|| "https://api.openai.com/v1".into());

    Ok(Some(EmbeddingsClient::new(
        &api_key,
        &base_url,
        &config.embedding_model,
        config.embedding_dimensions,
    )))
}
