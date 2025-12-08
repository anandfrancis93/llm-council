"""OpenRouter API client for making LLM requests."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


class ModelError:
    """Represents an error from a model query."""
    def __init__(self, model: str, error_type: str, message: str):
        self.model = model
        self.error_type = error_type  # 'timeout', 'api_error', 'rate_limit', 'network', 'unknown'
        self.message = message
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "model": self.model,
            "error_type": self.error_type,
            "message": self.message
        }


async def query_model_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,  # 2 minutes - these frontier models are slow
    max_retries: int = 2,
    base_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Query a single model via OpenRouter API with retry logic.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds (reduced from 180 to 60)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        Response dict with 'content', 'success', and optional 'error' fields
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    error = ModelError(model, "rate_limit", "Rate limited by API")
                    last_error = error
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        print(f"[{model}] Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                        await asyncio.sleep(delay)
                        continue
                    return {"success": False, "error": error.to_dict()}
                
                response.raise_for_status()
                data = response.json()
                
                # Check for valid response structure
                if 'choices' not in data or not data['choices']:
                    error = ModelError(model, "api_error", "Empty response from API")
                    return {"success": False, "error": error.to_dict()}
                
                message = data['choices'][0]['message']
                
                return {
                    "success": True,
                    "content": message.get('content'),
                    "reasoning_details": message.get('reasoning_details')
                }

        except httpx.TimeoutException:
            error = ModelError(model, "timeout", f"Request timed out after {timeout}s")
            last_error = error
            print(f"[{model}] Timeout (attempt {attempt + 1}/{max_retries + 1})")
            # Don't retry on timeout - likely the model is just slow
            return {"success": False, "error": error.to_dict()}
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.json()
                if 'error' in error_body:
                    error_msg = f"{error_msg}: {error_body['error'].get('message', str(error_body['error']))}"
            except:
                pass
            error = ModelError(model, "api_error", error_msg)
            last_error = error
            print(f"[{model}] API error: {error_msg} (attempt {attempt + 1}/{max_retries + 1})")
            
            # Retry on 5xx errors
            if e.response.status_code >= 500 and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                continue
            return {"success": False, "error": error.to_dict()}
            
        except httpx.ConnectError:
            error = ModelError(model, "network", "Failed to connect to API")
            last_error = error
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"[{model}] Connection error, retrying in {delay}s")
                await asyncio.sleep(delay)
                continue
            return {"success": False, "error": error.to_dict()}
            
        except Exception as e:
            error = ModelError(model, "unknown", str(e))
            last_error = error
            print(f"[{model}] Unexpected error: {e}")
            return {"success": False, "error": error.to_dict()}
    
    # Should not reach here, but just in case
    if last_error:
        return {"success": False, "error": last_error.to_dict()}
    return {"success": False, "error": ModelError(model, "unknown", "Max retries exceeded").to_dict()}


# Keep the old function signature for backward compatibility
async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 60.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API (backward compatible).
    
    Returns None on failure (legacy behavior), or dict with 'content' on success.
    """
    result = await query_model_with_retry(model, messages, timeout)
    if result.get("success"):
        return {
            'content': result.get('content'),
            'reasoning_details': result.get('reasoning_details')
        }
    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    timeout: float = 60.0
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel with detailed error tracking.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        timeout: Per-model timeout in seconds

    Returns:
        Dict mapping model identifier to response dict (includes success/error info)
    """
    # Create tasks for all models
    tasks = [query_model_with_retry(model, messages, timeout) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
