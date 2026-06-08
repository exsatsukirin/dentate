# Dentate 🧠

A lightweight, agent-native memory system — the dentate gyrus for your AI agents.

**No server. No local models. Just SQLite + API calls.**

Named after the **dentate gyrus**, the part of the hippocampus responsible for pattern separation — distinguishing similar memories from one another.

## Quick Start

### 1. Create config file

```bash
mkdir -p ~/.config/dentate
```

`~/.config/dentate/config.toml`:

```toml
[llm]
provider = "deepseek"
model = "deepseek-v4-flash"
api_key = "sk-xxx"
base_url = "https://api.deepseek.com/v1"

[embeddings]
# Use "dashscope" for 阿里百炼, "openai" for OpenAI, "none" to disable
provider = "dashscope"
model = "text-embedding-v3"
api_key = "sk-xxx"

[database]
path = "dentate.db"
```

### 2. Build

```bash
git clone https://github.com/exsatsukirin/dentate.git
cd dentate
cargo build --release
```

### 3. Use

```bash
# Store a memory
./target/release/dentate retain "Alice works at Google as a senior software engineer"

# Store with context
./target/release/dentate retain "The project deadline is next Friday" -c "team meeting notes"

# Search memories
./target/release/dentate recall "Where does Alice work?"

# Ask a question (searches + generates answer)
./target/release/dentate reflect "Summarize everything about Alice"

# Check stats
./target/release/dentate stats
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

All settings in `~/.config/dentate/config.toml`. CLI args and environment variables override config values.

### Provider Support

| Provider | LLM | Embeddings | China Access |
|----------|-----|------------|--------------|
| DeepSeek | ✅ | — | ✅ |
| DashScope (阿里百炼) | ✅ | ✅ | ✅ |
| Zhipu (智谱) | ✅ | ✅ | ✅ |
| OpenAI | ✅ | ✅ | ❌ (blocked) |
| None (keyword-only) | ✅ | — | — |

Recommended for China: **DeepSeek + DashScope**.

### Search Strategies

| Strategy | Description | Needs Embeddings |
|----------|-------------|-------------------|
| `hybrid` | Vector + FTS5 with RRF (default) | Yes |
| `semantic` | Pure vector similarity | Yes |
| `keyword` | Pure FTS5 full-text search | No |

### Embedding-Free Mode

Set `[embeddings] provider = "none"` to disable embeddings entirely:

- `retain()` skips embedding generation — stores only text + FTS5 index
- `recall()` uses keyword search only
- Database size drops ~90% (no vector storage)
- Most portable and sync-friendly mode

## Database Scale

Measured with 10K sample records:

```
                   Text + FTS5       With Embeddings (1024-dim)
Per record          ~520 bytes         ~4600 bytes  (4.6 KB)
1,000 records         0.5 MB             4.5 MB
10,000 records        5.0 MB            44 MB
50,000 records        25 MB            220 MB
```

The embedding vector dominates at 90% of storage. At typical usage (~2 records/day, 2,000 records after ~3 years), the database is ~9 MB with embeddings, ~1 MB without.

## Cross-Device Sync (Planned)

Current v0.1: SQLite single-file, copy = backup. Future sync approaches:

- **Export/import** (`dentate export` / `dentate import`) — recommended first step
- **File-level sync** (Syncthing / Dropbox) — single-writer only
- **WAL replication** (Litestream) — for continuous backup

> **Schema constraint**: the `vec_memories` table uses a fixed embedding dimension. Pin the same embedding provider across synced devices, or use `provider = "none"` for sync-friendly setups.

## License

MIT License. See [LICENSE](LICENSE) for details.
