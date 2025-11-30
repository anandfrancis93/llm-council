"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-4.1",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4",
    "x-ai/grok-3",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-2.5-flash"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
# Use /tmp on Vercel (serverless), local path otherwise
IS_VERCEL = os.getenv("VERCEL") == "1"
DATA_DIR = "/tmp/conversations" if IS_VERCEL else "data/conversations"
