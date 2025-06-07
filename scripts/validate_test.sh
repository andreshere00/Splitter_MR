#!/bin/bash
set -e

echo "🔍 Running test suite and checking for minimum 70% coverage..."

# Run tests with coverage
uv run coverage run --source=src -m pytest > /dev/null

# Show coverage report
uv run coverage report

# Check if coverage is above threshold; if not, print full report and fail
if ! uv run coverage report --fail-under=70 > /dev/null; then
    echo "❌ Coverage is below 70%."
    rm -f .coverage
    exit 1
fi

echo "✅ All tests pass and coverage is at or above 70%."

# Clean up the generated .coverage file
rm -f .coverage

exit 0
