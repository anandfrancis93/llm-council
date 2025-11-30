"""Vercel serverless adapter for the LLM Council API."""

import sys
import os

# Get the directory containing this file
api_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of api directory)
project_root = os.path.dirname(api_dir)

# Add project root to path so 'backend' package can be found
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the FastAPI app - this will trigger backend module initialization
from backend.main import app

# Vercel expects 'app' to be exported as the ASGI handler
