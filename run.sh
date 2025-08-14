#!/usr/bin/env bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 7860 &
uv run streamlit run frontend.py --server.headless true --server.port 8501
