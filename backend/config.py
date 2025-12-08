"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
# Using reliable, confirmed-working models
COUNCIL_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-1.5-pro",
    "meta-llama/llama-3.1-70b-instruct",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "anthropic/claude-3.5-sonnet"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Environment detection
IS_VERCEL = os.getenv("VERCEL") == "1"

# Redis/Upstash configuration for persistent storage
# Supports both Vercel KV and Upstash environment variable names
# Priority: KV_URL > UPSTASH_REDIS_REST_URL > KV_REST_API_URL

# Direct Redis URL (preferred - works with both Vercel KV and Upstash)
KV_URL = os.getenv("KV_URL") or os.getenv("UPSTASH_REDIS_REST_URL")

# REST API credentials (alternative method)
KV_REST_API_URL = os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN")

# Use Redis if any Redis URL is available, otherwise fallback to file storage
USE_REDIS = bool(KV_URL or KV_REST_API_URL)

# Data directory for file-based storage (local development fallback)
DATA_DIR = "data/conversations"
