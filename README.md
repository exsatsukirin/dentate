# Dentate 🧠

A lightweight, agent-native memory system — the dentate gyrus for your AI agents.

**No server. No local models. Just SQLite + API calls.**

Named after the **dentate gyrus**, the part of the hippocampus responsible for pattern separation — distinguishing similar memories from one another.

## Quick Start

### CLI

```bash
# Set your API key
export DEEPSEEK_API_KEY=sk-xxx
# or
export OPENAI_API_KEY=sk-xxx

# Store a memory
dentate retain "Alice works at Google as a senior software engineer"

# Store with context
dentate retain "The project deadline is next Friday" -c "team meeting notes"

# Search memories
dentate recall "Where does Alice work?"

# Ask a question (searches + generates answer)
dentate reflect "Summarize everything about Alice"

# Check stats
dentate stats
```

### Library

```rust
use dentate::MemoryBank;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bank = MemoryBank::open("agent_memory.db")?;

    // Store a memory (with automatic fact extraction)
    bank.retain("Alice works at Google as a software engineer", None, true).await?;

    // Recall relevant memories
    let results = bank.recall("Where does Alice work?", 5, Default::default()).await?;
    for m in &results.memories {
        println!("[{}] {} (score: {:.3})",
            &m.id[..8],
            m.fact.as_deref().unwrap_or(&m.content),
            m.relevance_score.unwrap_or(0.0)
        );
    }

    // Generate an answer from memories
    let answer = bank.reflect("Tell me about Alice").await?;
    println!("Answer: {}", answer.answer);

    Ok(())
}
```

## How It Works

```
retain("Alice works at Google")
    │
    ├── LLM extracts facts: ["Alice works at Google", "Alice is employed"]
    ├── API generates embeddings for each fact
    ├── Stores in SQLite (content + fact + embedding + FTS5 index)
    └── Returns stored memories

recall("Where does Alice work?")
    │
    ├── API generates query embedding
    ├── Vector search (semantic) ─────┐
    ├── FTS5 search (keyword) ────────┤── Reciprocal Rank Fusion
    └── Returns ranked memories  <────┘

reflect("Summarize Alice")
    │
    ├── recall(query) → retrieve relevant memories
    ├── LLM synthesizes answer from memories
    └── Returns answer + sources + token usage
```

## Configuration

All configuration through environment variables:

| Variable | Default | Description |
|---|---|---|
| `DENTATE_LLM_PROVIDER` | `deepseek` | LLM provider (deepseek, openai) |
| `DENTATE_LLM_MODEL` | `deepseek-chat` | LLM model name |
| `DENTATE_DATABASE` | `dentate.db` | SQLite database path |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback) |
| `COHERE_API_KEY` | — | Cohere API key (optional reranker) |

## Search Strategies

| Strategy | Description |
|---|---|
| `hybrid` | Vector + FTS5 with Reciprocal Rank Fusion (default) |
| `semantic` | Pure vector similarity search |
| `keyword` | Pure FTS5 full-text search |

## Cost

For a busy agent (100 retain + 500 recall + 50 reflect per day):
- Embeddings: ~$0.006/day
- LLM (fact extraction): ~$0.03/day
- LLM (reflection): ~$0.008/day
- **~$1.50/month** total

## License

MIT License. See [LICENSE](LICENSE) for details.
