#!/bin/bash
# Run integration tests that require API keys.
#
# Usage:
#   # Option 1: write key to ~/.dentate/api_key (one-time)
#   mkdir -p ~/.dentate
#   echo "sk-your-deepseek-key" > ~/.dentate/api_key
#   chmod 600 ~/.dentate/api_key
#   ./run_tests.sh
#
#   # Option 2: export directly
#   DEEPSEEK_API_KEY=sk-xxx ./run_tests.sh
#
#   # Option 3: pass specific test
#   ./run_tests.sh api_tests
#   ./run_tests.sh e2e_tests
#   ./run_tests.sh ""  # run all

set -euo pipefail

# Load key from file if env var not set
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    for keyfile in ~/.dentate/api_key tests/.api_key .api_key; do
        if [ -f "$keyfile" ]; then
            export DEEPSEEK_API_KEY=$(head -1 "$keyfile" | tr -d '\n')
            echo "Loaded API key from $keyfile"
            break
        fi
    done
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY,"
    echo "or create ~/.dentate/api_key with your key."
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
