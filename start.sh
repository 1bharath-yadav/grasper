#!/bin/bash

# Grasper API Startup Script
echo "🚀 Starting Grasper AI Data Analysis API..."

# Create necessary directories
mkdir -p logs

# Set environment variables
export ENVIRONMENT="development"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"


# Check if virtual environment is activated (uv)
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    exit 1
fi

echo "📦 Installing/updating dependencies with uv..."
uv sync

echo "🎭 Checking Playwright browsers..."
if ! uv run python -c "from playwright.sync_api import sync_playwright; sync_playwright().start().chromium.launch()" 2>/dev/null; then
    echo "Installing Playwright browsers..."
    uv run playwright install chromium
else
    echo "Playwright browsers already installed ✓"
fi


# Start the API with the complete workflow
uv run uvicorn  app.main:app --reload 



