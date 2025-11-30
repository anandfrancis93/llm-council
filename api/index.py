"""Vercel serverless adapter for the LLM Council API."""

import sys
from pathlib import Path

# Add the project root to the path so we can import the backend module
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app

# Vercel expects 'app' or 'handler' to be exported
handler = app
