//! Dentate CLI — command-line interface for the memory system.

use clap::{Parser, Subcommand};
use dentate::config;
use dentate::engine::MemoryBank;
use dentate::db::MemoryStore;
use dentate::types::{BankConfig, SearchStrategy};

#[derive(Parser)]
#[command(
    name = "dentate",
    about = "A lightweight, agent-native memory system — no server, no local models.",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Cli {
    /// Path to the SQLite database file (overrides config)
    #[arg(short, long, env = "DENTATE_DATABASE")]
    database: Option<String>,

    /// LLM provider: deepseek, openai (overrides config)
    #[arg(long, env = "DENTATE_LLM_PROVIDER")]
    llm_provider: Option<String>,

    /// LLM model name (overrides config)
    #[arg(long, env = "DENTATE_LLM_MODEL")]
    llm_model: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Store a new memory
    Retain {
        content: String,
        #[arg(short, long)]
        context: Option<String>,
        #[arg(long, default_value = "true")]
        extract_facts: bool,
    },
    /// Search for memories
    Recall {
        query: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
        #[arg(long, default_value = "hybrid")]
        strategy: String,
    },
    /// Generate an answer from memories
    Reflect {
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
    let cfg = config::load_or_default();

    let mut bank_cfg = BankConfig::from_config(&cfg);

    // CLI args override config
    if let Some(db) = cli.database { bank_cfg.database_path = db; }
    if let Some(p) = cli.llm_provider { bank_cfg.llm_provider = p; }
    if let Some(m) = cli.llm_model { bank_cfg.llm_model = m; }

    let bank = MemoryBank::with_config(bank_cfg)?;

    match cli.command {
        Command::Retain { content, context, extract_facts } => {
            tracing::info!("Retaining memory...");
            let memories = bank.retain(&content, context.as_deref(), extract_facts).await?;
            println!("Stored {} memories:", memories.len());
            for m in &memories {
                let fact = m.fact.as_deref().unwrap_or(&m.content);
                println!("  [{}] {}", &m.id[..8], fact);
            }
        }
        Command::Recall { query, limit, strategy } => {
            tracing::info!("Searching: {}", query);
            let strategy: SearchStrategy = strategy.parse().unwrap_or_default();
            let result = bank.recall(&query, limit, strategy).await?;
            println!("Found {} results (strategy: {:?}):", result.total_found, result.strategy_used);
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
