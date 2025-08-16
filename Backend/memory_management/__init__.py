"""
Memory Management System for Unified Agent Mind.

This package provides comprehensive memory storage and retrieval capabilities
for persona-based AI agents with support for:

- Memory chunk storage with metadata and tagging
- Interaction recording and retrieval
- Persona-specific memory management
- Search and filtering capabilities
- Rate limiting and authentication
- RESTful API with OpenAPI documentation

Main Components:
- models: Pydantic data models and schemas
- database: MongoDB integration and operations
- api: FastAPI endpoints and routing
- auth: Authentication and rate limiting
- utils: Helper functions and utilities

Usage:
    from memory_management.api import app
    from memory_management.database import get_memory_database
    from memory_management.models import MemoryCreateRequest
"""

__version__ = "1.0.0"
__author__ = "Gurukul Development Team"
__email__ = "dev@gurukul.com"

# Import main components for easy access
from .models import (
    MemoryCreateRequest,
    InteractionCreateRequest,
    MemoryUpdateRequest,
    MemoryChunkResponse,
    InteractionResponse,
    MemoryListResponse,
    InteractionListResponse,
    PersonaMemorySummary,
    MemorySearchResponse,
    ContentType,
    ImportanceLevel
)

from .database import get_memory_database, MemoryDatabase
from .api import app

__all__ = [
    # Models
    "MemoryCreateRequest",
    "InteractionCreateRequest", 
    "MemoryUpdateRequest",
    "MemoryChunkResponse",
    "InteractionResponse",
    "MemoryListResponse",
    "InteractionListResponse",
    "PersonaMemorySummary",
    "MemorySearchResponse",
    "ContentType",
    "ImportanceLevel",
    
    # Database
    "get_memory_database",
    "MemoryDatabase",
    
    # API
    "app"
]
