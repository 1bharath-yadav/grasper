#!/bin/bash

echo "📦 Installing dependencies with uv..."
uv sync

echo "🎭 Installing Playwright browsers..."
uv run playwright install chromium

echo "✅ Setup complete!"

# Start the application
uvicorn main:app --reload

# Docker deployment  
docker build -t data-analyst-agent .
docker run -p 8000:8000 data-analyst-agent