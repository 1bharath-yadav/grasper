
# *************** large language model configuration ***************

import os
from typing import Optional
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

FILES_DOWNLOADER_AGENT_MODEL = "gemini-2.5-flash"
ANSWER_AGENT_MODEL = "gemini-2.5-flash"
ORCHESTRATER_AGENT_MODEL = "gemini-2.5-flash"

# Initialize model
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    raise ValueError(
        "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set")

settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})
provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel("gemini-2.5-flash", provider=provider, settings=settings)

# Configuration functions


def get_gemini_model():
    """Get configured OpenAI model. Currently returns Gemini as fallback."""
    # TODO: Implement OpenAI model configuration when needed
    return model


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment variables."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return gemini_key
