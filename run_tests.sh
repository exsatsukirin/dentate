#!/bin/bash
# Run integration tests that require API keys.
#
# Keys are loaded from ~/.config/dentate/config.toml:
#
#   [llm]
#   api_key = "sk-xxx"
#
#   [embeddings]
#   api_key = "sk-xxx"      # defaults to llm.api_key
#
# Or set environment variables: DEEPSEEK_API_KEY, OPENAI_API_KEY, COHERE_API_KEY
#
# Usage:
#   ./run_tests.sh              # all tests
#   ./run_tests.sh api_tests    # specific test
#   ./run_tests.sh e2e_tests

set -euo pipefail

CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/dentate/config.toml"

load_key_from_toml() {
    local section="$1" key="$2"
    if [ -f "$CONFIG_FILE" ]; then
        awk -v sec="$section" -v k="$key" '
            $0 ~ "^\\[" sec "\\]" { in_sec=1; next }
            /^\[/ { in_sec=0 }
            in_sec && $1 == k {
                gsub(/^[^=]*= *"?/, "")
                gsub(/"$/, "")
                print
                exit
            }
        ' "$CONFIG_FILE"
    fi
}

# Load LLM key
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    KEY=$(load_key_from_toml "llm" "api_key")
    if [ -n "$KEY" ]; then
        export DEEPSEEK_API_KEY="$KEY"
        echo "Loaded DEEPSEEK_API_KEY from $CONFIG_FILE"
    fi
fi

# Load embeddings key independently (not gated on LLM key)
if [ -z "${OPENAI_API_KEY:-}" ]; then
    EMB_KEY=$(load_key_from_toml "embeddings" "api_key")
    if [ -n "$EMB_KEY" ]; then
        export OPENAI_API_KEY="$EMB_KEY"
        echo "Loaded OPENAI_API_KEY from $CONFIG_FILE"
    fi
fi

# Load Cohere key
if [ -z "${COHERE_API_KEY:-}" ]; then
    COH_KEY=$(load_key_from_toml "reranker" "api_key")
    if [ -n "$COH_KEY" ]; then
        export COHERE_API_KEY="$COH_KEY"
        echo "Loaded COHERE_API_KEY from $CONFIG_FILE"
    fi
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "No API key found."
    echo "Either create $CONFIG_FILE with [llm] api_key,"
    echo "or set DEEPSEEK_API_KEY / OPENAI_API_KEY environment variable."
    echo ""
    echo "Example $CONFIG_FILE:"
    echo '  [llm]'
    echo '  api_key = "sk-xxx"'
    exit 1
fi

TEST="${1:-}"
if [ -n "$TEST" ]; then
    echo "Running: cargo test --test $TEST -- --nocapture"
    cargo test --test "$TEST" -- --nocapture
else
    echo "Running: cargo test -- --nocapture"
    cargo test -- --nocapture
fi
