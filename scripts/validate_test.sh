#!/bin/bash
set -e

echo "🔍 Running test suite and checking for minimum 70% coverage..."

# Run tests with coverage directly (no uv)
uv run coverage run --source=src -m pytest

uv run coverage report

if ! uv run coverage report --fail-under=70 > /dev/null; then
    echo "❌ Coverage is below 70%."
    rm -f .coverage
    exit 1
fi

echo "✅ All tests pass and coverage is at or above 70%."

rm -f .coverage
exit 0
