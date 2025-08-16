"""
Pydantic models for the Memory Management System.

This module defines all data models, request/response schemas, and validation
rules for the memory management API endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from uuid import uuid4
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum


class ContentType(str, Enum):
    """Enumeration of memory content types."""
    TEXT = "text"
    INTERACTION = "interaction"
    CONTEXT = "context"
    REFLECTION = "reflection"
    PREFERENCE = "preference"
    FACT = "fact"


class ImportanceLevel(int, Enum):
    """Memory importance levels from 1 (low) to 10 (critical)."""
    VERY_LOW = 1
    LOW = 2
    BELOW_AVERAGE = 3
    AVERAGE = 4
    ABOVE_AVERAGE = 5
    MODERATE = 6
    HIGH = 7
    VERY_HIGH = 8
    CRITICAL = 9
    ESSENTIAL = 10


class MemoryMetadata(BaseModel):
    """Metadata associated with memory chunks."""
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    importance: ImportanceLevel = Field(default=ImportanceLevel.AVERAGE, description="Memory importance level")
    topic: Optional[str] = Field(None, description="Primary topic or subject")
    context_type: Optional[str] = Field(None, description="Type of context (conversation, task, etc.)")
    source: Optional[str] = Field(None, description="Source of the memory (user_input, agent_response, etc.)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the memory")
    related_memories: List[str] = Field(default_factory=list, description="IDs of related memory chunks")
    
    @validator('tags')
    def validate_tags(cls, v):
        """Ensure tags are non-empty strings."""
        return [tag.strip().lower() for tag in v if tag.strip()]


class InteractionContext(BaseModel):
    """Context information for interactions."""
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_turn: Optional[int] = Field(None, ge=1, description="Turn number in conversation")
    domain: Optional[str] = Field(None, description="Domain context (gurukul, finance, etc.)")
    intent: Optional[str] = Field(None, description="Detected user intent")
    previous_context: Optional[str] = Field(None, description="Previous conversation context")


class InteractionMetadata(BaseModel):
    """Metadata for interaction records."""
    response_time: Optional[float] = Field(None, ge=0.0, description="Response time in seconds")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence score")
    feedback: Optional[str] = Field(None, description="User feedback on the interaction")
    tags: List[str] = Field(default_factory=list, description="Interaction tags")
    model_used: Optional[str] = Field(None, description="AI model used for response")
    
    @validator('tags')
    def validate_tags(cls, v):
        """Ensure tags are non-empty strings."""
        return [tag.strip().lower() for tag in v if tag.strip()]


# Request Models

class MemoryCreateRequest(BaseModel):
    """Request model for creating a new memory chunk."""
    user_id: str = Field(..., description="User identifier")
    persona_id: str = Field(..., description="Persona identifier")
    content: str = Field(..., min_length=1, max_length=10000, description="Memory content")
    content_type: ContentType = Field(default=ContentType.TEXT, description="Type of memory content")
    metadata: Optional[MemoryMetadata] = Field(default_factory=MemoryMetadata, description="Memory metadata")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Memory timestamp")
    
    @validator('content')
    def validate_content(cls, v):
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or just whitespace")
        return v.strip()


class InteractionCreateRequest(BaseModel):
    """Request model for creating a new interaction record."""
    user_id: str = Field(..., description="User identifier")
    persona_id: str = Field(..., description="Persona identifier")
    user_message: str = Field(..., min_length=1, max_length=5000, description="User's message")
    agent_response: str = Field(..., min_length=1, max_length=10000, description="Agent's response")
    context: Optional[InteractionContext] = Field(default_factory=InteractionContext, description="Interaction context")
    metadata: Optional[InteractionMetadata] = Field(default_factory=InteractionMetadata, description="Interaction metadata")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Interaction timestamp")
    
    @validator('user_message', 'agent_response')
    def validate_messages(cls, v):
        """Ensure messages are not just whitespace."""
        if not v.strip():
            raise ValueError("Messages cannot be empty or just whitespace")
        return v.strip()


class MemoryUpdateRequest(BaseModel):
    """Request model for updating an existing memory chunk."""
    content: Optional[str] = Field(None, min_length=1, max_length=10000, description="Updated memory content")
    content_type: Optional[ContentType] = Field(None, description="Updated content type")
    metadata: Optional[MemoryMetadata] = Field(None, description="Updated metadata")
    is_active: Optional[bool] = Field(None, description="Whether memory is active")
    
    @validator('content')
    def validate_content(cls, v):
        """Ensure content is not just whitespace if provided."""
        if v is not None and not v.strip():
            raise ValueError("Content cannot be empty or just whitespace")
        return v.strip() if v else v


# Response Models

class MemoryChunkResponse(BaseModel):
    """Response model for memory chunk data."""
    memory_id: str = Field(..., description="Unique memory identifier")
    user_id: str = Field(..., description="User identifier")
    persona_id: str = Field(..., description="Persona identifier")
    content: str = Field(..., description="Memory content")
    content_type: ContentType = Field(..., description="Type of memory content")
    metadata: MemoryMetadata = Field(..., description="Memory metadata")
    timestamp: datetime = Field(..., description="Memory timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(..., description="Whether memory is active")


class InteractionResponse(BaseModel):
    """Response model for interaction data."""
    interaction_id: str = Field(..., description="Unique interaction identifier")
    user_id: str = Field(..., description="User identifier")
    persona_id: str = Field(..., description="Persona identifier")
    user_message: str = Field(..., description="User's message")
    agent_response: str = Field(..., description="Agent's response")
    context: InteractionContext = Field(..., description="Interaction context")
    metadata: InteractionMetadata = Field(..., description="Interaction metadata")
    timestamp: datetime = Field(..., description="Interaction timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_active: bool = Field(..., description="Whether interaction is active")


class MemoryListResponse(BaseModel):
    """Response model for paginated memory lists."""
    memories: List[MemoryChunkResponse] = Field(..., description="List of memory chunks")
    total_count: int = Field(..., ge=0, description="Total number of memories")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=500, description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class InteractionListResponse(BaseModel):
    """Response model for paginated interaction lists."""
    interactions: List[InteractionResponse] = Field(..., description="List of interactions")
    total_count: int = Field(..., ge=0, description="Total number of interactions")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=500, description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class PersonaMemorySummary(BaseModel):
    """Summary of memories for a specific persona."""
    persona_id: str = Field(..., description="Persona identifier")
    user_id: str = Field(..., description="User identifier")
    total_memories: int = Field(..., ge=0, description="Total number of memories")
    memory_categories: Dict[str, int] = Field(..., description="Count by content type")
    last_interaction: Optional[datetime] = Field(None, description="Last interaction timestamp")
    importance_distribution: Dict[str, int] = Field(..., description="Count by importance level")
    recent_topics: List[str] = Field(..., description="Recently discussed topics")
    updated_at: datetime = Field(..., description="Last update timestamp")


class MemorySearchResponse(BaseModel):
    """Response model for memory search results."""
    results: List[MemoryChunkResponse] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total number of results")
    search_time: float = Field(..., ge=0.0, description="Search execution time in seconds")
    suggestions: List[str] = Field(default_factory=list, description="Search suggestions")


# Error Response Models

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


# Success Response Models

class SuccessResponse(BaseModel):
    """Standard success response format."""
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class MemoryCreateResponse(SuccessResponse):
    """Response for successful memory creation."""
    data: Dict[str, str] = Field(..., description="Created memory information")
    
    @model_validator(mode='before')
    @classmethod
    def set_memory_data(cls, values):
        """Ensure data contains memory_id."""
        if isinstance(values, dict):
            if 'data' not in values or not values['data'].get('memory_id'):
                values['data'] = {'memory_id': str(uuid4())}
        return values


class InteractionCreateResponse(SuccessResponse):
    """Response for successful interaction creation."""
    data: Dict[str, str] = Field(..., description="Created interaction information")
    
    @model_validator(mode='before')
    @classmethod
    def set_interaction_data(cls, values):
        """Ensure data contains interaction_id."""
        if isinstance(values, dict):
            if 'data' not in values or not values['data'].get('interaction_id'):
                values['data'] = {'interaction_id': str(uuid4())}
        return values
