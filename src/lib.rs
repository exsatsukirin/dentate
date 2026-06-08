//! Dentate — A lightweight, agent-native memory system.
//!
//! Named after the dentate gyrus, the part of the hippocampus responsible for
//! pattern separation — distinguishing similar memories from one another.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use dentate::MemoryBank;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let bank = MemoryBank::open("agent_memory.db")?;
//!
//!     // Store a memory
//!     bank.retain("Alice works at Google as a software engineer", None, true).await?;
//!
//!     // Recall memories
//!     let results = bank.recall("Where does Alice work?", 5, Default::default()).await?;
//!     for m in &results.memories {
//!         println!("{} (score: {:.3})", m.content, m.relevance_score.unwrap_or(0.0));
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod db;
pub mod engine;
pub mod error;
pub mod types;

// Re-export the main types for convenience
pub use engine::MemoryBank;
pub use types::*;
