//! Dentate CLI — command-line interface for the memory system.

use clap::{Parser, Subcommand};
use dentate::engine::MemoryBank;
use dentate::db::MemoryStore;
use dentate::types::SearchStrategy;

#[derive(Parser)]
#[command(
    name = "dentate",
    about = "A lightweight, agent-native memory system — no server, no local models.",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Cli {
    /// Path to the SQLite database file
    #[arg(short, long, default_value = "dentate.db", env = "DENTATE_DATABASE")]
    database: String,

    /// LLM provider (deepseek, openai)
    #[arg(long, default_value = "deepseek", env = "DENTATE_LLM_PROVIDER")]
    llm_provider: String,

    /// LLM model name
    #[arg(long, default_value = "deepseek-chat", env = "DENTATE_LLM_MODEL")]
    llm_model: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Store a new memory
    Retain {
        /// The content to remember
        content: String,

        /// Optional context for the memory
        #[arg(short, long)]
        context: Option<String>,

        /// Extract atomic facts using LLM (default: true)
        #[arg(long, default_value = "true")]
        extract_facts: bool,
    },

    /// Search for memories
    Recall {
        /// The search query
        query: String,

        /// Maximum results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Search strategy: hybrid, semantic, or keyword
        #[arg(long, default_value = "hybrid")]
        strategy: String,
    },

    /// Generate an answer from memories
    Reflect {
        /// The question to answer
        query: String,
    },

    /// Show database statistics
    Stats,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "dentate=info".into()),
        )
        .init();

    let cli = Cli::parse();

    let config = dentate::types::BankConfig {
        database_path: cli.database,
        llm_provider: cli.llm_provider,
        llm_model: cli.llm_model,
        ..Default::default()
    };

    let bank = MemoryBank::with_config(config)?;

    match cli.command {
        Command::Retain {
            content,
            context,
            extract_facts,
        } => {
            tracing::info!("Retaining memory...");
            let memories = bank
                .retain(&content, context.as_deref(), extract_facts)
                .await?;
            println!("Stored {} memories:", memories.len());
            for m in &memories {
                let fact = m.fact.as_deref().unwrap_or(&m.content);
                println!("  [{}] {}", &m.id[..8], fact);
            }
        }

        Command::Recall {
            query,
            limit,
            strategy,
        } => {
            tracing::info!("Searching: {}", query);
            let strategy = SearchStrategy::from_str(&strategy);
            let result = bank.recall(&query, limit, strategy).await?;
            println!(
                "Found {} results (strategy: {:?}):",
                result.total_found, result.strategy_used
            );
            for m in &result.memories {
                let score = m.relevance_score.unwrap_or(0.0);
                let fact = m.fact.as_deref().unwrap_or(&m.content);
                println!("  [{:.4}] {}", score, fact);
            }
        }

        Command::Reflect { query } => {
            tracing::info!("Reflecting: {}", query);
            let result = bank.reflect(&query).await?;
            println!("{}", result.answer);
            println!("\n--- Sources ({}) ---", result.sources.len());
            for m in &result.sources {
                let fact = m.fact.as_deref().unwrap_or(&m.content);
                println!("  • {}", fact);
            }
            println!(
                "\nToken usage: {} in / {} out",
                result.token_usage.input_tokens, result.token_usage.output_tokens
            );
        }

        Command::Stats => {
            let db = bank.db().lock().await;
            let count = MemoryStore::count(&db)?;
            println!("Database: {}", bank.config().database_path);
            println!("Total memories: {}", count);
        }
    }

    Ok(())
}
