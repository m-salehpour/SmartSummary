#!/bin/bash
set -e  # exit immediately on error

echo "🔹 Cleaning Python cache files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -r {} +

echo "🔹 Formatting with Black..."
black .

echo "🔹 Sorting imports with isort..."
isort .

echo "🔹 Linting with Ruff..."
ruff check . --fix

echo "🔹 Type checking with mypy..."
mypy .

echo "🔹 Freezing dependencies..."
pip freeze > requirements.txt

echo "🔹 Running tests..."
pytest

echo "✅ All checks passed. Ready to push!"
