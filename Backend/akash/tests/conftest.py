"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
import os
from datetime import datetime, timedelta
from jose import jwt

# Set test environment variables
os.environ["SUPABASE_JWT_SECRET"] = "test-secret-key"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["MONGODB_URI"] = "mongodb://localhost:27017/"
os.environ["MONGODB_DATABASE"] = "test_agent_memory"
os.environ["AGENT_API_URL"] = "http://localhost:8000/ask-agent"
os.environ["AGENT_API_KEY"] = "test-api-key"

from main import app
from memory.mongodb_client import MongoDBClient
from api.chat_handler import ChatHandler

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def mock_mongodb():
    """Mock MongoDB client"""
    mock_client = AsyncMock(spec=MongoDBClient)
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_client.save_chat_message = AsyncMock(return_value="test-message-id")
    mock_client.get_user_chat_history = AsyncMock(return_value=[])
    mock_client.save_user_session = AsyncMock(return_value="test-session-id")
    mock_client.get_user_session = AsyncMock(return_value=None)
    mock_client.delete_user_data = AsyncMock(return_value={"chat_messages": 0, "sessions": 0})
    return mock_client

@pytest.fixture
def mock_chat_handler():
    """Mock chat handler"""
    mock_handler = Mock(spec=ChatHandler)
    mock_handler.validate_message = Mock(return_value=True)
    mock_handler.call_agent_api = AsyncMock(return_value={
        "success": True,
        "response": "Test response from agent",
        "metadata": {"model": "test-model", "tokens": 100}
    })
    mock_handler.generate_tts_audio = AsyncMock(return_value=None)
    return mock_handler

@pytest.fixture
def valid_jwt_token():
    """Generate a valid JWT token for testing"""
    payload = {
        "sub": "test-user-123",
        "email": "test@example.com",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "aud": "authenticated"
    }
    return jwt.encode(payload, "test-secret-key", algorithm="HS256")

@pytest.fixture
def expired_jwt_token():
    """Generate an expired JWT token for testing"""
    payload = {
        "sub": "test-user-123",
        "email": "test@example.com",
        "exp": datetime.utcnow() - timedelta(hours=1),
        "aud": "authenticated"
    }
    return jwt.encode(payload, "test-secret-key", algorithm="HS256")

@pytest.fixture
def invalid_jwt_token():
    """Generate an invalid JWT token for testing"""
    return "invalid.jwt.token"

@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing"""
    return [
        {
            "_id": "msg1",
            "user_id": "test-user-123",
            "message": "Hello",
            "response": "Hi there!",
            "timestamp": datetime.utcnow() - timedelta(minutes=5),
            "metadata": {}
        },
        {
            "_id": "msg2",
            "user_id": "test-user-123",
            "message": "How are you?",
            "response": "I'm doing well, thank you!",
            "timestamp": datetime.utcnow() - timedelta(minutes=2),
            "metadata": {}
        }
    ]
