# app/config/settings.py
"""
Application settings and configuration management.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Directories
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
TEMP_DIR = os.getenv("TEMP_DIR", "tmp")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
