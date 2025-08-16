"""
Basic tests for the Memory Management API.

Run with: python -m pytest test_api.py -v
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient

from .api import app
from .models import MemoryCreateRequest, InteractionCreateRequest, ContentType, ImportanceLevel

# Test client
client = TestClient(app)

# Test data
TEST_API_KEY = "memory_api_key_test"
TEST_HEADERS = {
    "Authorization": f"Bearer {TEST_API_KEY}",
    "Content-Type": "application/json"
}

TEST_USER_ID = "test_user_123"
TEST_PERSONA_ID = "test_persona"


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/memory/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestAuthentication:
    """Test authentication and authorization."""
    
    def test_missing_auth_header(self):
        """Test request without authorization header."""
        response = client.post("/memory", json={})
        assert response.status_code == 401
    
    def test_invalid_api_key(self):
        """Test request with invalid API key."""
        headers = {"Authorization": "Bearer invalid_key"}
        response = client.post("/memory", json={}, headers=headers)
        assert response.status_code == 401


class TestMemoryStorage:
    """Test memory storage endpoints."""
    
    def test_create_memory_success(self):
        """Test successful memory creation."""
        memory_data = {
            "user_id": TEST_USER_ID,
            "persona_id": TEST_PERSONA_ID,
            "content": "Test memory content",
            "content_type": "text",
            "metadata": {
                "tags": ["test", "example"],
                "importance": 5,
                "topic": "testing"
            }
        }
        
        response = client.post("/memory", json=memory_data, headers=TEST_HEADERS)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert "memory_id" in data["data"]
        assert data["message"] == "Memory chunk created successfully"
    
    def test_create_memory_validation_error(self):
        """Test memory creation with validation errors."""
        # Missing required fields
        memory_data = {
            "user_id": TEST_USER_ID,
            # Missing persona_id and content
        }
        
        response = client.post("/memory", json=memory_data, headers=TEST_HEADERS)
        assert response.status_code == 422  # Validation error
    
    def test_create_interaction_success(self):
        """Test successful interaction creation."""
        interaction_data = {
            "user_id": TEST_USER_ID,
            "persona_id": TEST_PERSONA_ID,
            "user_message": "Test user message",
            "agent_response": "Test agent response",
            "context": {
                "session_id": "test_session",
                "conversation_turn": 1,
                "domain": "test"
            },
            "metadata": {
                "response_time": 1.0,
                "confidence": 0.9
            }
        }
        
        response = client.post("/memory/interaction", json=interaction_data, headers=TEST_HEADERS)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert "interaction_id" in data["data"]


class TestMemoryRetrieval:
    """Test memory retrieval endpoints."""
    
    def test_get_memories_by_persona(self):
        """Test retrieving memories by persona."""
        # First create a memory
        memory_data = {
            "user_id": TEST_USER_ID,
            "persona_id": TEST_PERSONA_ID,
            "content": "Test memory for retrieval",
            "content_type": "text"
        }
        
        create_response = client.post("/memory", json=memory_data, headers=TEST_HEADERS)
        assert create_response.status_code == 201
        
        # Then retrieve memories
        response = client.get(
            f"/memory?persona={TEST_PERSONA_ID}&limit=10",
            headers=TEST_HEADERS
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "memories" in data
        assert "total_count" in data
        assert "page" in data
    
    def test_get_recent_interactions(self):
        """Test retrieving recent interactions."""
        # First create an interaction
        interaction_data = {
            "user_id": TEST_USER_ID,
            "persona_id": TEST_PERSONA_ID,
            "user_message": "Test message for retrieval",
            "agent_response": "Test response for retrieval"
        }
        
        create_response = client.post("/memory/interaction", json=interaction_data, headers=TEST_HEADERS)
        assert create_response.status_code == 201
        
        # Then retrieve interactions
        response = client.get(
            "/memory?limit=5&recent_interactions=true",
            headers=TEST_HEADERS
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "interactions" in data
        assert "total_count" in data
    
    def test_get_memories_missing_persona(self):
        """Test retrieving memories without persona parameter."""
        response = client.get("/memory?limit=10", headers=TEST_HEADERS)
        assert response.status_code == 400  # Bad request


class TestMemorySearch:
    """Test memory search functionality."""
    
    def test_search_memories(self):
        """Test memory search."""
        # First create a searchable memory
        memory_data = {
            "user_id": TEST_USER_ID,
            "persona_id": TEST_PERSONA_ID,
            "content": "This is a searchable test memory about investments",
            "content_type": "text",
            "metadata": {
                "tags": ["investment", "test"],
                "topic": "finance"
            }
        }
        
        create_response = client.post("/memory", json=memory_data, headers=TEST_HEADERS)
        assert create_response.status_code == 201
        
        # Then search for it
        response = client.get(
            "/memory/search?query=investment&limit=5",
            headers=TEST_HEADERS
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "total_results" in data
        assert "search_time" in data
    
    def test_search_memories_missing_query(self):
        """Test search without query parameter."""
        response = client.get("/memory/search", headers=TEST_HEADERS)
        assert response.status_code == 422  # Validation error


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_memory_create_request_validation(self):
        """Test MemoryCreateRequest validation."""
        # Valid request
        valid_data = {
            "user_id": "test_user",
            "persona_id": "test_persona",
            "content": "Test content",
            "content_type": ContentType.TEXT
        }
        
        request = MemoryCreateRequest(**valid_data)
        assert request.user_id == "test_user"
        assert request.content == "Test content"
        assert request.content_type == ContentType.TEXT
        
        # Invalid request - empty content
        with pytest.raises(ValueError):
            MemoryCreateRequest(
                user_id="test_user",
                persona_id="test_persona",
                content="",  # Empty content should fail
                content_type=ContentType.TEXT
            )
    
    def test_interaction_create_request_validation(self):
        """Test InteractionCreateRequest validation."""
        # Valid request
        valid_data = {
            "user_id": "test_user",
            "persona_id": "test_persona",
            "user_message": "Test user message",
            "agent_response": "Test agent response"
        }
        
        request = InteractionCreateRequest(**valid_data)
        assert request.user_message == "Test user message"
        assert request.agent_response == "Test agent response"
        
        # Invalid request - empty messages
        with pytest.raises(ValueError):
            InteractionCreateRequest(
                user_id="test_user",
                persona_id="test_persona",
                user_message="",  # Empty message should fail
                agent_response="Test response"
            )


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_memory_id(self):
        """Test retrieving non-existent memory."""
        response = client.get("/memory/invalid_id", headers=TEST_HEADERS)
        assert response.status_code == 404
    
    def test_invalid_persona_summary(self):
        """Test retrieving summary for non-existent persona."""
        response = client.get("/memory/persona/non_existent/summary", headers=TEST_HEADERS)
        assert response.status_code == 404


# Integration test
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow: create, retrieve, search, update, delete."""
    
    # This would be a more comprehensive integration test
    # that tests the full workflow of the API
    
    # 1. Create memory
    memory_data = {
        "user_id": TEST_USER_ID,
        "persona_id": TEST_PERSONA_ID,
        "content": "Integration test memory",
        "content_type": "text",
        "metadata": {
            "tags": ["integration", "test"],
            "importance": 6,
            "topic": "testing"
        }
    }
    
    create_response = client.post("/memory", json=memory_data, headers=TEST_HEADERS)
    assert create_response.status_code == 201
    
    memory_id = create_response.json()["data"]["memory_id"]
    
    # 2. Retrieve memory
    get_response = client.get(f"/memory/{memory_id}", headers=TEST_HEADERS)
    assert get_response.status_code == 200
    
    retrieved_memory = get_response.json()
    assert retrieved_memory["content"] == "Integration test memory"
    
    # 3. Search for memory
    search_response = client.get(
        "/memory/search?query=integration&limit=5",
        headers=TEST_HEADERS
    )
    assert search_response.status_code == 200
    
    search_results = search_response.json()
    assert search_results["total_results"] >= 1
    
    # 4. Update memory
    update_data = {
        "content": "Updated integration test memory",
        "is_active": True
    }
    
    update_response = client.put(f"/memory/{memory_id}", json=update_data, headers=TEST_HEADERS)
    assert update_response.status_code == 200
    
    # 5. Delete memory (soft delete)
    delete_response = client.delete(f"/memory/{memory_id}", headers=TEST_HEADERS)
    assert delete_response.status_code == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
