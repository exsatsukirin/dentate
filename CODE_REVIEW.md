# Dentate Code Review — 2026-06-08

## Critical (should fix now)

### 1. API tests hardcode model name instead of reading from config
**Files**: `tests/api_tests.rs:2318,2347,2367`
All three LLM tests hardcode `LlmClient::new(..., "deepseek-chat")` instead of using 
the model from the user's config. If config says `deepseek-v4-flash`, tests ignore it.
**Fix**: Use `config::load_or_default()` to get the model name.

### 2. Dead code: `require_embeddings_key` never called
**File**: `tests/api_tests.rs:2285`
Function defined but replaced by `make_embeddings_client()`. Adds a compiler warning.
**Fix**: Remove the dead function.

### 3. `Default for BankConfig` has stale relative path
**File**: `src/types.rs:2249`
`database_path` defaults to `"dentate.db"` (relative) but should use XDG path.
`embedding_provider` defaults to `"openai"` but should match config defaults.
`llm_model` defaults to `"deepseek-chat"` which is not a valid DeepSeek model name.
**Fix**: Update Default impl to match `config.rs` defaults.

### 4. `DentateError` enum never used
**File**: `src/error.rs`
Entire custom error type is unused — everything uses `anyhow::Result`. Either use it
or remove it to reduce dead code.
**Fix**: Use `DentateError` in API clients for specific error variants, or remove the module.

## Medium (should fix soon)

### 5. No retry/backoff for transient API failures
Both `embeddings.rs` and `llm.rs` fail immediately on any non-2xx response. Rate limits
and transient network errors should be retried.
**Fix**: Add 2-3 retries with exponential backoff for 429/5xx responses.

### 6. Column-index-based row access in MemoryStore
`MemoryStore::get()` uses `row.get(0)`, `row.get(1)` etc. Fragile to schema changes.
**Fix**: Use named column access: `row.get("id")`, `row.get("content")`.

### 7. No `updated_at` updates
Schema has `updated_at` column but nothing writes to it. Either implement UPDATE 
support or remove the column.
**Fix**: Add `MemoryStore::update()` or remove the column from schema.

### 8. `test_resolve_api_key_env_first` misnamed
Test name says "env_first" but actually tests config fallback (env var doesn't exist).
**Fix**: Rename to `test_resolve_api_key_fallback_to_config`.

## Minor (nice to have)

### 9. No `.dockerignore` or CI config
Project has no CI/CD pipeline. GitHub Actions for `cargo test --test db_tests` would 
be useful.
**Fix**: Add `.github/workflows/test.yml`.

### 10. Schema comment references stale OpenAI dimensions
`schema.rs` has "For OpenAI text-embedding-3-small, change to float[1536]" but 
the actual default provider is now dashscope with 1024 dims.
**Fix**: Update comment to reflect current defaults.

### 11. No `rustfmt.toml` or `.editorconfig`
Consistent formatting across contributors would benefit from a config file.
**Fix**: Add `rustfmt.toml` with `max_width = 100`.

## Already Good
- ✅ Tests skip gracefully when API keys missing
- ✅ Provider abstraction (dashscope/zhipu/openai/none) clean and extensible
- ✅ RRF fusion implementation correct and tested
- ✅ WAL mode enabled by default
- ✅ Config file follows XDG spec
- ✅ Binary size (~6.8MB) is excellent for Rust
- ✅ README comprehensive and up-to-date
