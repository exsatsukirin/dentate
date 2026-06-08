//! Error types for the Dentate memory system.

/// Errors that can occur in Dentate operations.
#[derive(Debug, thiserror::Error)]
pub enum DentateError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("API error: {0}")]
    Api(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}
