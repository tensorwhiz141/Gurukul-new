"""
Utility functions for the Memory Management System.

This module provides helper functions for data formatting, pagination,
validation, and other common operations.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from .models import (
    MemoryChunkResponse, InteractionResponse, PersonaMemorySummary,
    ContentType, ImportanceLevel, MemoryMetadata, InteractionContext,
    InteractionMetadata
)

# Configure logging
logger = logging.getLogger(__name__)


def format_memory_response(memory_data: Dict[str, Any]) -> MemoryChunkResponse:
    """
    Format raw memory data from database into response model.
    
    Args:
        memory_data: Raw memory data from MongoDB
        
    Returns:
        MemoryChunkResponse: Formatted memory response
    """
    try:
        # Extract metadata
        metadata_dict = memory_data.get("metadata", {})
        metadata = MemoryMetadata(
            tags=metadata_dict.get("tags", []),
            importance=ImportanceLevel(metadata_dict.get("importance", 4)),
            topic=metadata_dict.get("topic"),
            context_type=metadata_dict.get("context_type"),
            source=metadata_dict.get("source"),
            confidence=metadata_dict.get("confidence"),
            related_memories=metadata_dict.get("related_memories", [])
        )
        
        return MemoryChunkResponse(
            memory_id=memory_data["memory_id"],
            user_id=memory_data["user_id"],
            persona_id=memory_data["persona_id"],
            content=memory_data["content"],
            content_type=ContentType(memory_data["content_type"]),
            metadata=metadata,
            timestamp=memory_data["timestamp"],
            created_at=memory_data["created_at"],
            updated_at=memory_data["updated_at"],
            is_active=memory_data["is_active"]
        )
        
    except Exception as e:
        logger.error(f"Error formatting memory response: {e}")
        raise ValueError(f"Invalid memory data format: {e}")


def format_interaction_response(interaction_data: Dict[str, Any]) -> InteractionResponse:
    """
    Format raw interaction data from database into response model.
    
    Args:
        interaction_data: Raw interaction data from MongoDB
        
    Returns:
        InteractionResponse: Formatted interaction response
    """
    try:
        # Extract context
        context_dict = interaction_data.get("context", {})
        context = InteractionContext(
            session_id=context_dict.get("session_id"),
            conversation_turn=context_dict.get("conversation_turn"),
            domain=context_dict.get("domain"),
            intent=context_dict.get("intent"),
            previous_context=context_dict.get("previous_context")
        )
        
        # Extract metadata
        metadata_dict = interaction_data.get("metadata", {})
        metadata = InteractionMetadata(
            response_time=metadata_dict.get("response_time"),
            confidence=metadata_dict.get("confidence"),
            feedback=metadata_dict.get("feedback"),
            tags=metadata_dict.get("tags", []),
            model_used=metadata_dict.get("model_used")
        )
        
        return InteractionResponse(
            interaction_id=interaction_data["interaction_id"],
            user_id=interaction_data["user_id"],
            persona_id=interaction_data["persona_id"],
            user_message=interaction_data["user_message"],
            agent_response=interaction_data["agent_response"],
            context=context,
            metadata=metadata,
            timestamp=interaction_data["timestamp"],
            created_at=interaction_data["created_at"],
            is_active=interaction_data["is_active"]
        )
        
    except Exception as e:
        logger.error(f"Error formatting interaction response: {e}")
        raise ValueError(f"Invalid interaction data format: {e}")


def paginate_results(
    results: List[Any],
    total_count: int,
    page: int,
    page_size: int
) -> Dict[str, Any]:
    """
    Create pagination information for results.
    
    Args:
        results: List of results
        total_count: Total number of available results
        page: Current page number (1-based)
        page_size: Number of items per page
        
    Returns:
        Dict with pagination information
    """
    total_pages = (total_count + page_size - 1) // page_size
    
    return {
        "results": results,
        "pagination": {
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
            "next_page": page + 1 if page < total_pages else None,
            "previous_page": page - 1 if page > 1 else None
        }
    }


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate if a string is a valid UUID.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        bool: True if valid UUID
    """
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


def sanitize_content(content: str) -> str:
    """
    Sanitize content by removing potentially harmful characters.
    
    Args:
        content: Content to sanitize
        
    Returns:
        str: Sanitized content
    """
    # Remove control characters except newlines and tabs
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    # Limit consecutive whitespace
    content = re.sub(r'\s{3,}', '  ', content)
    
    # Trim whitespace
    content = content.strip()
    
    return content


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for tagging.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction - can be enhanced with NLP libraries
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stop words and count frequency
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]


def calculate_content_similarity(content1: str, content2: str) -> float:
    """
    Calculate similarity between two content strings.
    
    Args:
        content1: First content string
        content2: Second content string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Simple Jaccard similarity - can be enhanced with embeddings
    words1 = set(re.findall(r'\b\w+\b', content1.lower()))
    words2 = set(re.findall(r'\b\w+\b', content2.lower()))
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for consistent display.
    
    Args:
        timestamp: Datetime object
        
    Returns:
        str: Formatted timestamp string
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def validate_persona_id(persona_id: str) -> bool:
    """
    Validate persona ID format.
    
    Args:
        persona_id: Persona identifier
        
    Returns:
        bool: True if valid
    """
    # Allow alphanumeric, hyphens, underscores
    pattern = r'^[a-zA-Z0-9_-]{1,50}$'
    return bool(re.match(pattern, persona_id))


def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format.
    
    Args:
        user_id: User identifier
        
    Returns:
        bool: True if valid
    """
    # Allow alphanumeric, hyphens, underscores, dots
    pattern = r'^[a-zA-Z0-9._-]{1,50}$'
    return bool(re.match(pattern, user_id))


def create_memory_summary(memories: List[Dict[str, Any]]) -> str:
    """
    Create a summary of memories for context.
    
    Args:
        memories: List of memory dictionaries
        
    Returns:
        str: Summary text
    """
    if not memories:
        return "No memories available."
    
    # Group by content type
    content_groups = {}
    for memory in memories:
        content_type = memory.get("content_type", "text")
        if content_type not in content_groups:
            content_groups[content_type] = []
        content_groups[content_type].append(memory)
    
    summary_parts = []
    
    for content_type, group_memories in content_groups.items():
        count = len(group_memories)
        recent_content = group_memories[-3:]  # Last 3 memories
        
        summary_parts.append(f"{content_type.title()} memories ({count}):")
        for memory in recent_content:
            content = memory.get("content", "")[:100]  # First 100 chars
            timestamp = memory.get("timestamp", "")
            summary_parts.append(f"  - {content}... ({timestamp})")
    
    return "\n".join(summary_parts)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text (rough approximation).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        int: Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_content(content: str, max_tokens: int = 1000) -> str:
    """
    Truncate content to fit within token limit.
    
    Args:
        content: Content to truncate
        max_tokens: Maximum token count
        
    Returns:
        str: Truncated content
    """
    max_chars = max_tokens * 4  # Rough approximation
    
    if len(content) <= max_chars:
        return content
    
    # Truncate at word boundary
    truncated = content[:max_chars]
    last_space = truncated.rfind(' ')
    
    if last_space > max_chars * 0.8:  # If we can find a space reasonably close
        truncated = truncated[:last_space]
    
    return truncated + "..."
