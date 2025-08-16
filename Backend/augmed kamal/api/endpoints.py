"""
API Endpoints for Agent Mind-Auth-Memory Link
Protected routes for chat and progress saving
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Dict, Any

from auth.supabase_auth import get_current_user
from memory.mongodb_client import MongoDBClient
from api.chat_handler import ChatHandler
from models.schemas import (
    ChatRequest, ChatResponse, SaveProgressRequest, SaveProgressResponse,
    UserInfo, ErrorResponse
)

# Create router
router = APIRouter()

# Initialize components (these will be injected in main.py)
mongodb_client = None
chat_handler = None

def set_dependencies(mongo_client: MongoDBClient, chat_handler_instance: ChatHandler):
    """Set the dependencies for the endpoints"""
    global mongodb_client, chat_handler
    mongodb_client = mongo_client
    chat_handler = chat_handler_instance

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Protected chat endpoint that integrates with Vedant's agent API
    
    - Verifies user authentication
    - Retrieves chat history if requested
    - Calls agent API with context
    - Saves the exchange to MongoDB
    - Optionally generates TTS audio
    """
    try:
        user_id = current_user["user_id"]
        
        # Validate the message
        if not chat_handler.validate_message(request.message):
            raise HTTPException(
                status_code=400,
                detail="Invalid message: empty or too long"
            )
        
        # Get chat history if requested
        chat_history = []
        if request.include_history:
            chat_history = await mongodb_client.get_user_chat_history(
                user_id=user_id,
                limit=request.max_history_messages
            )
        
        # Call the agent API
        agent_response = await chat_handler.call_agent_api(
            user_message=request.message,
            chat_history=chat_history,
            user_id=user_id
        )
        
        if not agent_response["success"]:
            # If agent API failed, return the fallback response
            response_text = agent_response["response"]
            metadata = {"error": agent_response.get("error", "Unknown error")}
        else:
            response_text = agent_response["response"]
            metadata = agent_response.get("metadata", {})
        
        # Generate TTS audio if requested
        audio_url = None
        if request.tts_enabled:
            audio_url = await chat_handler.generate_tts_audio(response_text, user_id)
            if audio_url:
                metadata["tts_generated"] = True
                metadata["audio_url"] = audio_url
        
        # Save the chat exchange to MongoDB
        try:
            await mongodb_client.save_chat_message(
                user_id=user_id,
                message=request.message,
                response=response_text,
                metadata=metadata
            )
        except Exception as e:
            # Log the error but don't fail the request
            # The user still gets their response even if saving fails
            metadata["save_error"] = str(e)
        
        return ChatResponse(
            response=response_text,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            audio_url=audio_url,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing error: {str(e)}"
        )

@router.post("/save_progress", response_model=SaveProgressResponse)
async def save_progress_endpoint(
    request: SaveProgressRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Protected endpoint to save user's current chat session
    
    - Verifies user authentication
    - Saves session data to MongoDB
    - Returns confirmation
    """
    try:
        user_id = current_user["user_id"]
        
        # Convert Pydantic models to dictionaries for MongoDB
        chat_history_dicts = [msg.model_dump() for msg in request.chat_history]
        
        # Save the session
        session_id = await mongodb_client.save_user_session(
            user_id=user_id,
            chat_history=chat_history_dicts,
            session_metadata=request.session_metadata
        )
        
        return SaveProgressResponse(
            success=True,
            user_id=user_id,
            saved_messages=len(request.chat_history),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save progress: {str(e)}"
        )

@router.get("/chat_history")
async def get_chat_history(
    limit: int = 50,
    skip: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get user's chat history
    
    - Verifies user authentication
    - Returns paginated chat history
    """
    try:
        user_id = current_user["user_id"]
        
        history = await mongodb_client.get_user_chat_history(
            user_id=user_id,
            limit=limit,
            skip=skip
        )
        
        return {
            "user_id": user_id,
            "chat_history": history,
            "count": len(history),
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )

@router.get("/user_session")
async def get_user_session(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get user's current session
    
    - Verifies user authentication
    - Returns current session data
    """
    try:
        user_id = current_user["user_id"]
        
        session = await mongodb_client.get_user_session(user_id)
        
        if not session:
            return {
                "user_id": user_id,
                "session": None,
                "message": "No active session found"
            }
        
        return {
            "user_id": user_id,
            "session": session
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user session: {str(e)}"
        )

@router.delete("/user_data")
async def delete_user_data(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete all user data (GDPR compliance)
    
    - Verifies user authentication
    - Deletes all chat history and sessions
    """
    try:
        user_id = current_user["user_id"]
        
        deleted_counts = await mongodb_client.delete_user_data(user_id)
        
        return {
            "user_id": user_id,
            "deleted": deleted_counts,
            "message": "All user data has been deleted"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete user data: {str(e)}"
        )
