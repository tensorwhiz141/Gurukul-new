"""
Agent Mind-Auth-Memory Link
Main FastAPI application that binds voice (TTS), identity (Supabase), and memory (MongoDB)
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

from auth.supabase_auth import verify_token, get_current_user
from memory.mongodb_client import MongoDBClient
from api.chat_handler import ChatHandler
from api.endpoints import router as api_router, set_dependencies
from models.schemas import ChatRequest, ChatResponse, SaveProgressRequest

# Load environment variables
load_dotenv()

# Initialize components
mongodb_client = MongoDBClient()
chat_handler = ChatHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    try:
        await mongodb_client.connect()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("   Application will continue without MongoDB (limited functionality)")

    # Set dependencies for API endpoints
    set_dependencies(mongodb_client, chat_handler)
    yield

    # Shutdown
    try:
        await mongodb_client.close()
    except Exception as e:
        print(f"Warning: Error closing MongoDB connection: {e}")

app = FastAPI(
    title="Agent Mind-Auth-Memory Link",
    description="AI Agent with Authentication and Memory",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up API routes
app.include_router(api_router, prefix="/api/v1", tags=["Agent API"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agent Mind-Auth-Memory Link is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "mongodb": await mongodb_client.health_check(),
        "components": ["auth", "memory", "chat"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
