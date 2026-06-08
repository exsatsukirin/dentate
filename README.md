# Dentate

> A lightweight, agent-native memory system — the dentate gyrus for your AI agents.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org/)

**No server. No local models. Just SQLite + API calls.**

Named after the [dentate gyrus](https://en.wikipedia.org/wiki/Dentate_gyrus), the part of the hippocampus responsible for pattern separation — distinguishing similar memories from one another.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [Library](#library)
- [Search Strategies](#search-strategies)
- [Embedding-Free Mode](#embedding-free-mode)
- [Database](#database)
- [Roadmap](#roadmap)
- [Development](#development)
- [License](#license)

---

## Features

- **Hybrid search** — Vector similarity + FTS5 full-text search with Reciprocal Rank Fusion
- **Automatic fact extraction** — LLM splits raw text into atomic, self-contained facts
- **Multi-provider** — DeepSeek, DashScope (阿里百炼), Zhipu (智谱), OpenAI, or keyword-only
- **Embedding-free mode** — Disable embeddings for maximum portability and smaller databases
- **Single-file database** — One SQLite file, easy to back up or sync
- **Agent-native** — Library and CLI designed for programmatic use by AI agents
- **Minimal footprint** — ~7 MB binary, no server process, no GPU required

## Architecture

```
retain("Alice works at Google")
    │
    ├── LLM extracts facts ─── ["Alice works at Google", "Alice is employed"]
    ├── Embedding API ───────── generates vector for each fact
    ├── SQLite ──────────────── stores content + fact + embedding + FTS5 index
    └── Returns stored memories

recall("Where does Alice work?")
    │
    ├── Embedding API ───────── generates query vector
    ├── sqlite-vec ──────────── vector similarity search
    ├── FTS5 ────────────────── keyword full-text search
    ├── RRF ─────────────────── merges and re-ranks results
    └── Returns ranked memories

reflect("Summarize Alice")
    │
    ├── recall(query) ──────── retrieve relevant memories
    ├── LLM ────────────────── synthesizes answer from memories
    └── Returns answer + sources + token usage
```

## Installation

### Prerequisites

- Rust 1.87+ ([rustup](https://rustup.rs/))
- An LLM API key (DeepSeek, OpenAI, etc.)
- An embeddings API key (optional, for semantic search)

### Build from source

```bash
git clone https://github.com/exsatsukirin/dentate.git
cd dentate
cargo build --release
```

The binary will be at `./target/release/dentate`.

### Verify installation

```bash
./target/release/dentate --version
```

## Configuration

Dentate reads configuration from `~/.config/dentate/config.toml`. All fields are optional — missing values fall back to built-in defaults or environment variables.

### Config file

```toml
[llm]
provider = "deepseek"          # "deepseek", "openai", "dashscope", "zhipu"
model = "deepseek-chat"        # model name for your provider
api_key = "sk-xxx"             # or set DEEPSEEK_API_KEY / OPENAI_API_KEY env var
# base_url = "https://api.deepseek.com/v1"  # optional, auto-detected from provider

[embeddings]
provider = "dashscope"         # "dashscope", "openai", "zhipu", "none"
model = "text-embedding-v3"    # embedding model name
api_key = "sk-xxx"             # or set OPENAI_API_KEY env var
# dimensions = 1024            # optional, auto-detected from provider

[reranker]
enabled = false                # enable Cohere reranker (optional)
# api_key = "sk-xxx"           # or set COHERE_API_KEY env var

[database]
path = "~/.config/dentate/dentate.db"  # default path (XDG compliant)
```

### Environment variables

Environment variables take precedence over config file values:

| Variable | Purpose |
|----------|---------|
| `DENTATE_LLM_API_KEY` | LLM API key (highest priority) |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENAI_API_KEY` | OpenAI API key (also used for embeddings) |
| `DENTATE_EMBEDDINGS_API_KEY` | Embeddings API key (overrides OPENAI_API_KEY) |
| `COHERE_API_KEY` | Cohere reranker API key |

### Provider support

| Provider | LLM | Embeddings |
|----------|:---:|:----------:|
| DeepSeek | ✅ | — |
| DashScope (阿里百炼) | ✅ | ✅ |
| Zhipu (智谱) | ✅ | ✅ |
| OpenAI | ✅ | ✅ |
| None (keyword-only) | ✅ | — |

**Recommended:** DeepSeek (LLM) + DashScope (embeddings).

## Usage

### CLI

```bash
# Store a memory (automatic fact extraction)
dentate retain "Alice works at Google as a senior software engineer"

# Store with context
dentate retain "The project deadline is next Friday" -c "team meeting notes"

# Store without fact extraction (raw text)
dentate retain "Raw text stored as-is" --extract-facts false

# Search memories
dentate recall "Where does Alice work?"

# Search with specific strategy
dentate recall "Alice" --strategy keyword --limit 5

# Generate an answer from memories
dentate reflect "Summarize everything about Alice"

# List all memories
dentate list

# List with search filter
dentate list "Alice"

# Show database statistics
dentate stats
```

### Library

Add to your `Cargo.toml`:

```toml
[dependencies]
dentate = { git = "https://github.com/exsatsukirin/dentate" }
```

#### Basic usage

```rust
use dentate::MemoryBank;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Open or create a memory bank (reads ~/.config/dentate/config.toml)
    let bank = MemoryBank::open("agent_memory.db")?;

    // Store a memory with automatic fact extraction
    bank.retain(
        "Alice works at Google as a software engineer",
        None,      // optional context
        true,      // extract facts via LLM
    ).await?;

    // Recall relevant memories
    let results = bank.recall(
        "Where does Alice work?",
        5,                          // max results
        Default::default(),         // hybrid search
    ).await?;

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
    println!("Tokens: {} in / {} out",
        answer.token_usage.input_tokens,
        answer.token_usage.output_tokens
    );

    Ok(())
}
```

#### Custom metadata

```rust
use serde_json::json;

let memories = bank.retain_with_metadata(
    "The API rate limit is 100 req/min",
    Some("production notes"),
    true,
    json!({ "source": "ops-meeting", "priority": "high" }),
).await?;
```

#### Custom configuration

```rust
use dentate::types::BankConfig;

let config = BankConfig {
    database_path: "my_agent.db".into(),
    llm_model: "deepseek-v4-flash".into(),
    embedding_provider: "none".into(),  // keyword-only mode
    ..Default::default()
};
let bank = MemoryBank::with_config(config)?;
```

## Search Strategies

| Strategy | Description | Requires Embeddings |
|----------|-------------|:-------------------:|
| `hybrid` | Vector similarity + FTS5, merged with Reciprocal Rank Fusion | Yes |
| `semantic` | Pure vector similarity search | Yes |
| `keyword` | Pure FTS5 full-text search | No |

The default strategy is `hybrid`. When embeddings are disabled, all strategies automatically fall back to `keyword`.

### Reciprocal Rank Fusion

Dentate uses [Reciprocal Rank Fusion (RRF)](https://research.google/pubs/reciprocal-rank-fusion-beyond-pairwise-rank-fusion/) to combine vector and keyword search results. RRF scores each document by summing `1/(k + rank)` across all result lists, where `k=60` (a smoothing constant). Documents appearing in both lists naturally rank higher.

## Embedding-Free Mode

Set `provider = "none"` in the `[embeddings]` section to disable vector search entirely:

```toml
[embeddings]
provider = "none"
```

Effects:
- `retain()` skips embedding generation — stores only text + FTS5 index
- `recall()` uses keyword search only (FTS5)
- Database size drops ~90% (no vector storage)
- No external embedding API calls
- Most portable and sync-friendly mode

## Database

Dentate uses a single SQLite file with WAL mode enabled by default.

### Schema

| Table | Purpose |
|-------|---------|
| `memories` | Core memory storage (id, content, fact, context, metadata, created_at) |
| `memories_fts` | FTS5 full-text search index (auto-synced via triggers) |
| `vec_memories` | Vector embeddings (sqlite-vec virtual table) |
| `dentate_meta` | Internal metadata (embedding dimension, etc.) |

### Storage estimates

Measured with 10K sample records (1024-dim embeddings):

| Records | Text + FTS5 | With Embeddings |
|--------:|------------:|----------------:|
| 1,000 | 0.5 MB | 4.5 MB |
| 10,000 | 5.0 MB | 44 MB |
| 50,000 | 25 MB | 220 MB |

At typical usage (~2 records/day), expect ~9 MB after 3 years with embeddings, ~1 MB without.

### Backup

The database is a single file. To back up:

```bash
# While dentate is not running:
cp ~/.config/dentate/dentate.db ~/backup/dentate-$(date +%Y%m%d).db

# Or use SQLite's backup command for live backups:
sqlite3 ~/.config/dentate/dentate.db ".backup ~/backup/dentate.db"
```

### Dimension constraint

The embedding dimension is fixed at database creation time. If you switch embedding providers (e.g., DashScope 1024-dim → OpenAI 1536-dim), you must delete and recreate the database, or Dentate will report a dimension mismatch error.

## Roadmap

- [ ] Export/import commands (`dentate export` / `dentate import`)
- [ ] Memory update and delete via CLI
- [ ] Configurable embedding dimension per database
- [ ] Cross-device sync (file-level or WAL replication)
- [ ] Batch retain from file
- [ ] Memory deduplication

## Development

### Running tests

```bash
# Unit tests (no API keys needed)
cargo test --lib --test db_tests

# Integration tests (requires API keys in config or env vars)
cargo test --test api_tests --test e2e_tests

# All tests
cargo test
```

### Project structure

```
src/
├── api/                    # External API clients
│   ├── embeddings.rs       #   Embedding generation (OpenAI-compatible)
│   ├── llm.rs              #   LLM inference (fact extraction, reflection)
│   └── reranker.rs         #   Cohere reranker (optional)
├── bin/
│   └── dentate.rs          #   CLI entry point
├── db/                     # Database layer
│   ├── memories.rs         #   CRUD operations
│   ├── schema.rs           #   Migrations
│   └── search.rs           #   FTS5 + vector search + RRF
├── engine/                 # Core pipeline
│   ├── retain.rs           #   Store memories
│   ├── recall.rs           #   Search memories
│   └── reflect.rs          #   Generate answers
├── config.rs               # Configuration loading
├── types.rs                # Data types and config structs
└── lib.rs                  # Public API
```

## License

MIT License. See [LICENSE](LICENSE) for details.
