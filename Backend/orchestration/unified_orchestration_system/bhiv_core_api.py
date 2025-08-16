"""
BHIV Core API
FastAPI endpoints for the BHIV Core Orchestrator Service
Provides the main API interface for the BHIV Core schema implementation
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Local imports
from bhiv_core_schema import (
    BHIVCoreRequest, BHIVCoreResponse, UserInfo, UserInput,
    ClassificationType, UrgencyLevel, TriggerModule, OutputFormat,
    create_bhiv_request, create_agent_directives
)
from bhiv_core_orchestrator import BHIVCoreOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global orchestrator instance
bhiv_orchestrator: Optional[BHIVCoreOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global bhiv_orchestrator
    
    # Startup
    logger.info("Starting BHIV Core API...")
    bhiv_orchestrator = BHIVCoreOrchestrator()
    await bhiv_orchestrator.initialize()
    logger.info("BHIV Core API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BHIV Core API...")


# Create FastAPI app
app = FastAPI(
    title="BHIV Core API",
    description="BHIV Core Orchestrator Service - Manages the complete flow: USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO",
    version="0.9.0",
    lifespan=lifespan
)

# Static files (audio) setup
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator() -> BHIVCoreOrchestrator:
    """Dependency to get orchestrator instance"""
    if bhiv_orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BHIV Core Orchestrator not initialized"
        )
    return bhiv_orchestrator


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "BHIV Core API",
        "version": "0.9.0",
        "status": "running",
        "description": "BHIV Core Orchestrator Service",
        "flow": "USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO"
    }


@app.get("/health")
async def health_check(orchestrator: BHIVCoreOrchestrator = Depends(get_orchestrator)):
    """Health check endpoint"""
    try:
        health_info = await orchestrator.health_check()
        return health_info
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/api/v1/bhiv-core/process", response_model=BHIVCoreResponse)
async def process_bhiv_request(
    request: BHIVCoreRequest,
    background_tasks: BackgroundTasks,
    orchestrator: BHIVCoreOrchestrator = Depends(get_orchestrator)
):
    """
    Main BHIV Core processing endpoint
    Processes a complete BHIV Core request through the full pipeline
    """
    try:
        logger.info(f"Processing BHIV Core request for user: {request.user_info.user_id}")
        
        # Process the request through the full BHIV pipeline
        response = await orchestrator.process_bhiv_request(request)
        
        # Log successful processing
        background_tasks.add_task(
            log_interaction,
            request.session_id,
            request.user_info.user_id,
            request.input.raw_text,
            response.output.response_text,
            response.logs.reward_score
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing BHIV request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process BHIV request: {str(e)}"
        )


class QuickQueryRequest(BaseModel):
    user_id: str
    query: str
    voice_enabled: bool = False
    lang: str = "en-IN"


@app.post("/api/v1/bhiv-core/quick-query")
async def quick_query(
    req: QuickQueryRequest,
    orchestrator: BHIVCoreOrchestrator = Depends(get_orchestrator)
):
    """
    Quick query endpoint for simple requests
    Creates a BHIV request with sensible defaults
    """
    try:
        # Create BHIV request with defaults
        request = create_bhiv_request(
            user_id=req.user_id,
            raw_text=req.query,
            voice_enabled=req.voice_enabled,
            lang=req.lang
        )
        
        # Process the request
        response = await orchestrator.process_bhiv_request(request)
        
        # Return simplified response
        return {
            "session_id": response.session_id,
            "response_text": response.output.response_text,
            "confidence_score": response.output.confidence_score,
            "voice_response_url": response.output.voice_response_url,
            "followup_suggestions": response.output.followup_suggestions,
            "agent_used": response.agent_directives.agent_name,
            "classification": response.agent_directives.classification,
            "processing_time_ms": response.logs.processing_time_ms,
            "sources": response.output.sources[:3]  # Limit sources for quick response
        }
        
    except Exception as e:
        logger.error(f"Error processing quick query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process quick query: {str(e)}"
        )


@app.get("/api/v1/bhiv-core/session/{session_id}")
async def get_session_info(
    session_id: str,
    orchestrator: BHIVCoreOrchestrator = Depends(get_orchestrator)
):
    """Get session information"""
    try:
        session_info = await orchestrator.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session info: {str(e)}"
        )


class ClassifyRequest(BaseModel):
    user_id: str
    query: str
    lang: str = "en-IN"


@app.post("/api/v1/bhiv-core/classify")
async def classify_query(
    req: ClassifyRequest,
    orchestrator: BHIVCoreOrchestrator = Depends(get_orchestrator)
):
    """
    Classify a query using BHIV Alpha router
    Returns classification without full processing
    """
    try:
        # Create user input
        user_input = UserInput(raw_text=req.query)
        user_info = UserInfo(user_id=req.user_id, auth_token="temp", lang=req.lang)
        
        # Classify the query
        classification = await orchestrator.alpha_router.classify_query(user_input, user_info)
        
        return {
            "classification": classification.classification,
            "confidence": classification.confidence,
            "urgency": classification.urgency,
            "recommended_agent": classification.recommended_agent,
            "reasoning": classification.reasoning
        }
        
    except Exception as e:
        logger.error(f"Error classifying query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to classify query: {str(e)}"
        )


@app.get("/api/v1/bhiv-core/stats")
async def get_stats(orchestrator: BHIVCoreOrchestrator = Depends(get_orchestrator)):
    """Get BHIV Core statistics"""
    try:
        stats = {
            "total_sessions": len(orchestrator.session_store),
            "active_sessions": len([s for s in orchestrator.session_store.values() if s.get('status') == 'processing']),
            "completed_sessions": len([s for s in orchestrator.session_store.values() if s.get('status') == 'completed']),
            "error_sessions": len([s for s in orchestrator.session_store.values() if s.get('status') == 'error']),
            "components_status": {
                "orchestration_engine": "ready" if orchestrator.orchestration_engine else "not_ready",
                "gemini_manager": "available" if orchestrator.gemini_manager.is_available() else "unavailable",
                "knowledgebase": "ready" if orchestrator.knowledgebase.vector_stores else "not_ready",
                "voice_processor": "ready"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


async def log_interaction(
    session_id: str,
    user_id: str,
    query: str,
    response: str,
    reward_score: float
):
    """Background task to log interactions"""
    try:
        # This could be enhanced to log to a database or external service
        logger.info(f"Interaction logged - Session: {session_id}, User: {user_id}, Reward: {reward_score}")
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bhiv_core_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8010")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
