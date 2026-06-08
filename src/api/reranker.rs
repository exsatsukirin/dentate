//! Reranker API client (Cohere Rerank compatible).

/// Client for re-ranking search results using Cohere's Rerank API.
///
/// This is optional — by default, vector similarity scores are used directly.
pub struct RerankerClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl RerankerClient {
    /// Create a new reranker client.
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Rerank documents against a query.
    ///
    /// Returns (doc_index, relevance_score) pairs, sorted best-first.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        });

        let response = self
            .client
            .post("https://api.cohere.com/v2/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Cohere rerank API error ({}): {}", status, text);
        }

        let result: serde_json::Value = response.json().await?;
        let results = result["results"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid rerank response"))?;

        let ranked: Vec<(usize, f32)> = results
            .iter()
            .map(|r| {
                let idx = r["index"].as_u64().unwrap_or(0) as usize;
                let score = r["relevance_score"].as_f64().unwrap_or(0.0) as f32;
                (idx, score)
            })
            .collect();

        Ok(ranked)
    }
}
