"""Storage for conversations - uses Upstash Redis in production, JSON files locally."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR, USE_REDIS, KV_REST_API_URL, KV_REST_API_TOKEN

# Redis client (only initialized if Redis is available)
_redis_client = None


def _get_redis():
    """Get or create Upstash Redis client."""
    global _redis_client
    if _redis_client is None:
        from upstash_redis import Redis
        # Initialize with REST API URL and token from environment
        _redis_client = Redis(url=KV_REST_API_URL, token=KV_REST_API_TOKEN)
    return _redis_client


# =============================================================================
# Redis Storage Implementation (Upstash)
# =============================================================================

def _redis_create_conversation(conversation_id: str) -> Dict[str, Any]:
    """Create a new conversation in Redis."""
    r = _get_redis()
    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    # Store conversation as JSON string with key pattern: conversation:{id}
    r.set(f"conversation:{conversation_id}", json.dumps(conversation))
    # Add to the set of all conversation IDs for listing
    r.sadd("conversation_ids", conversation_id)
    return conversation


def _redis_get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Load a conversation from Redis."""
    r = _get_redis()
    data = r.get(f"conversation:{conversation_id}")
    if data is None:
        return None
    # upstash-redis returns the value directly as string
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    return json.loads(data)


def _redis_save_conversation(conversation: Dict[str, Any]):
    """Save a conversation to Redis."""
    r = _get_redis()
    r.set(f"conversation:{conversation['id']}", json.dumps(conversation))


def _redis_list_conversations() -> List[Dict[str, Any]]:
    """List all conversations from Redis (metadata only)."""
    r = _get_redis()
    conversation_ids = r.smembers("conversation_ids")
    
    if not conversation_ids:
        return []
    
    conversations = []
    for cid in conversation_ids:
        # Handle bytes or string
        if isinstance(cid, bytes):
            cid = cid.decode('utf-8')
        
        data = r.get(f"conversation:{cid}")
        if data:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            conv = json.loads(data)
            conversations.append({
                "id": conv["id"],
                "created_at": conv["created_at"],
                "message_count": len(conv["messages"])
            })
    
    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    return conversations


def _redis_delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation from Redis."""
    r = _get_redis()
    # Remove from the set of conversation IDs
    r.srem("conversation_ids", conversation_id)
    # Delete the conversation data
    deleted = r.delete(f"conversation:{conversation_id}")
    return deleted > 0


# =============================================================================
# File Storage Implementation (Local Development)
# =============================================================================

def _ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def _get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def _file_create_conversation(conversation_id: str) -> Dict[str, Any]:
    """Create a new conversation in file storage."""
    _ensure_data_dir()
    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    path = _get_conversation_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)
    return conversation


def _file_get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Load a conversation from file storage."""
    path = _get_conversation_path(conversation_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def _file_save_conversation(conversation: Dict[str, Any]):
    """Save a conversation to file storage."""
    _ensure_data_dir()
    path = _get_conversation_path(conversation['id'])
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)


def _file_list_conversations() -> List[Dict[str, Any]]:
    """List all conversations from file storage (metadata only)."""
    _ensure_data_dir()
    conversations = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "message_count": len(data["messages"])
                })
    
    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    return conversations


def _file_delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation from file storage."""
    path = _get_conversation_path(conversation_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# =============================================================================
# Public API - Routes to appropriate storage backend
# =============================================================================

def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    if USE_REDIS:
        return _redis_create_conversation(conversation_id)
    return _file_create_conversation(conversation_id)


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    if USE_REDIS:
        return _redis_get_conversation(conversation_id)
    return _file_get_conversation(conversation_id)


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    if USE_REDIS:
        return _redis_save_conversation(conversation)
    return _file_save_conversation(conversation)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    if USE_REDIS:
        return _redis_list_conversations()
    return _file_list_conversations()


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        True if deleted, False if not found
    """
    if USE_REDIS:
        return _redis_delete_conversation(conversation_id)
    return _file_delete_conversation(conversation_id)


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "user",
        "content": content
    })

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any]
):
    """
    Add an assistant message with all 3 stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage3: Final synthesized response
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3
    })

    save_conversation(conversation)
