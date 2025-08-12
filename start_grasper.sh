#!/bin/bash

# Run uvicorn and log output
uv run uvicorn app.main:app --reload --port 8000 

# API_PID=$!

# # Start the Streamlit frontend  
# echo "🎨 Starting frontend on port 8501..."
 uv run streamlit run frontend.py --server.headless true --server.port 8501 &
# FRONTEND_PID=$!

