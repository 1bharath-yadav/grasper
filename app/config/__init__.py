# app/config/__init__.py
"""
Main configuration module for Grasper AI Data Analysis API.
This module exposes all configuration functions and settings.
"""

from .llm_models import (
    model,
    FILES_DOWNLOADER_AGENT_MODEL,
    ANSWER_AGENT_MODEL,
    ORCHESTRATER_AGENT_MODEL,
    get_gemini_model,
    get_gemini_model,
    get_gemini_api_key,
    get_gemini_api_key,
)

from .settings import (
    API_HOST,
    API_PORT,
    API_RELOAD,
    ENVIRONMENT,
    LOG_LEVEL,
    CACHE_DIR,
    TEMP_DIR,
    LOGS_DIR,
)

__all__ = [
    # LLM Models
    "model",
    "FILES_DOWNLOADER_AGENT_MODEL",
    "ANSWER_AGENT_MODEL",
    "ORCHESTRATER_AGENT_MODEL",
    "get_gemini_model",
    "get_gemini_model",
    "get_gemini_api_key",
    "get_gemini_api_key",
    # Settings
    "API_HOST",
    "API_PORT",
    "API_RELOAD",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "CACHE_DIR",
    "TEMP_DIR",
    "LOGS_DIR",
]
