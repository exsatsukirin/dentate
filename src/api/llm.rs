//! LLM API client for fact extraction and reflection.

use serde::{Deserialize, Serialize};

/// Client for LLM inference (fact extraction, reflection).
pub struct LlmClient {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl LlmClient {
    /// Create a new LLM client.
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Extract atomic facts from a piece of content using the LLM.
    pub async fn extract_facts(
        &self,
        content: &str,
        context: Option<&str>,
    ) -> anyhow::Result<Vec<String>> {
        let system_prompt = concat!(
            "You are a fact extraction system. Extract all atomic, self-contained facts from the input. ",
            "Each fact should be a single, complete statement that can stand alone. ",
            r#"Return ONLY a JSON array of strings, nothing else. "#,
            r#"Example: ["Alice works at Google", "Alice is a software engineer", "Alice lives in Beijing"]"#
        );

        let user_prompt = if let Some(ctx) = context {
            format!("Context: {}\n\nContent: {}", ctx, content)
        } else {
            content.to_string()
        };

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt.into(),
                },
                Message {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            temperature: 0.3,
            max_tokens: Some(1024),
        };

        let text = self.chat(&request).await?;

        // Parse JSON array from response (strip markdown code blocks if present)
        let cleaned = text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        let facts: Vec<String> = serde_json::from_str(cleaned)?;
        Ok(facts)
    }

    /// Generate a reflection (answer) based on retrieved memories.
    pub async fn reflect(
        &self,
        query: &str,
        memories: &[crate::types::Memory],
    ) -> anyhow::Result<(String, crate::types::TokenUsage)> {
        let memories_text: String = memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let fact_text = m.fact.as_deref().unwrap_or(&m.content);
                format!("[{}] {}", i + 1, fact_text)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let system_prompt = concat!(
            "You are a helpful assistant with access to the user's memory bank. ",
            "Answer the user's question based ONLY on the provided memories. ",
            "If the memories don't contain enough information, say so honestly. ",
            "Cite memory numbers [1], [2], etc. when making claims."
        );

        let user_prompt = format!("Memories:\n{}\n\nQuestion: {}", memories_text, query);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt.into(),
                },
                Message {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            temperature: 0.7,
            max_tokens: Some(1024),
        };

        let (answer, usage) = self.chat_with_usage(&request).await?;
        Ok((answer, usage))
    }

    /// Send a chat request and return just the text response.
    async fn chat(&self, request: &ChatRequest) -> anyhow::Result<String> {
        let (text, _) = self.chat_with_usage(request).await?;
        Ok(text)
    }

    /// Send a chat request and return both text and token usage.
    async fn chat_with_usage(
        &self,
        request: &ChatRequest,
    ) -> anyhow::Result<(String, crate::types::TokenUsage)> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API error ({}): {}", status, text);
        }

        let result: ChatResponse = response.json().await?;

        let answer = result
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = result
            .usage
            .map(|u| crate::types::TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            })
            .unwrap_or_default();

        Ok((answer, usage))
    }
}
