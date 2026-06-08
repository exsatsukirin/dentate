//! Embeddings API client using reqwest (OpenAI-compatible).

use serde::{Deserialize, Serialize};

/// Client for generating text embeddings via OpenAI-compatible API.
pub struct EmbeddingsClient {
    api_key: String,
    base_url: String,
    model: String,
    dimensions: Option<u32>,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
}

impl EmbeddingsClient {
    /// Create a new embeddings client.
    pub fn new(api_key: &str, base_url: &str, model: &str, dimensions: Option<u32>) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dimensions,
            client: reqwest::Client::new(),
        }
    }

    /// Generate an embedding vector for a single text.
    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no embedding returned"))
    }

    /// Generate embeddings for multiple texts in a batch.
    pub async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);

        let input = if texts.len() == 1 {
            serde_json::Value::String(texts[0].to_string())
        } else {
            serde_json::Value::Array(texts.iter().map(|s| serde_json::Value::String(s.to_string())).collect())
        };

        let mut request = EmbeddingRequest {
            model: self.model.clone(),
            input,
            dimensions: self.dimensions,
        };

        // DeepSeek doesn't have embeddings API — but for other info:
        // if someone sets a deepseek model, strip dimensions
        if self.model.contains("deepseek") {
            request.dimensions = None;
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API error ({}): {}", status, text);
        }

        let result: EmbeddingResponse = response.json().await?;

        let embeddings: Vec<Vec<f32>> = result
            .data
            .into_iter()
            .map(|d| d.embedding.into_iter().map(|v| v as f32).collect())
            .collect();

        if embeddings.len() != texts.len() {
            anyhow::bail!(
                "expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            );
        }

        Ok(embeddings)
    }
}
