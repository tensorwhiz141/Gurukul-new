"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")

class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    message: str = Field(..., description="User's message to the agent")
    include_history: bool = Field(default=True, description="Whether to include chat history")
    max_history_messages: int = Field(default=10, description="Maximum number of history messages to include")
    tts_enabled: bool = Field(default=False, description="Whether to generate TTS audio")

class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    response: str = Field(..., description="Agent's response")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    audio_url: Optional[str] = Field(default=None, description="TTS audio URL if enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")

class SaveProgressRequest(BaseModel):
    """Request model for /save_progress endpoint"""
    chat_history: List[ChatMessage] = Field(..., description="Current session's chat history")
    session_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional session metadata")

class SaveProgressResponse(BaseModel):
    """Response model for /save_progress endpoint"""
    success: bool = Field(..., description="Whether the save operation was successful")
    user_id: str = Field(..., description="User identifier")
    saved_messages: int = Field(..., description="Number of messages saved")
    timestamp: datetime = Field(..., description="Save timestamp")

class UserInfo(BaseModel):
    """User information from token"""
    user_id: str = Field(..., description="User identifier")
    email: Optional[str] = Field(default=None, description="User email")
    exp: Optional[int] = Field(default=None, description="Token expiration timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    mongodb: Dict[str, Any] = Field(..., description="MongoDB health status")
    components: List[str] = Field(..., description="Available components")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
