"""
Tests for API endpoints
"""

import pytest
from unittest.mock import patch, AsyncMock
import json
from datetime import datetime

class TestChatEndpoint:
    
    def test_chat_endpoint_success(self, client, valid_jwt_token, mock_mongodb, mock_chat_handler):
        """Test successful chat endpoint call"""
        with patch('api.endpoints.mongodb_client', mock_mongodb), \
             patch('api.endpoints.chat_handler', mock_chat_handler):
            
            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Hello, how are you?",
                    "include_history": True,
                    "max_history_messages": 10,
                    "tts_enabled": False
                },
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["response"] == "Test response from agent"
            assert data["user_id"] == "test-user-123"
            assert "timestamp" in data
            assert data["audio_url"] is None
            assert "metadata" in data
    
    def test_chat_endpoint_with_tts(self, client, valid_jwt_token, mock_mongodb, mock_chat_handler):
        """Test chat endpoint with TTS enabled"""
        mock_chat_handler.generate_tts_audio.return_value = "http://example.com/audio.mp3"
        
        with patch('api.endpoints.mongodb_client', mock_mongodb), \
             patch('api.endpoints.chat_handler', mock_chat_handler):
            
            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Hello",
                    "tts_enabled": True
                },
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["audio_url"] == "http://example.com/audio.mp3"
            assert data["metadata"]["tts_generated"] is True
    
    def test_chat_endpoint_unauthorized(self, client):
        """Test chat endpoint without authorization"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 401
    
    def test_chat_endpoint_invalid_message(self, client, valid_jwt_token, mock_mongodb, mock_chat_handler):
        """Test chat endpoint with invalid message"""
        mock_chat_handler.validate_message.return_value = False
        
        with patch('api.endpoints.mongodb_client', mock_mongodb), \
             patch('api.endpoints.chat_handler', mock_chat_handler):
            
            response = client.post(
                "/api/v1/chat",
                json={"message": ""},
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 400
            assert "invalid message" in response.json()["detail"].lower()
    
    def test_chat_endpoint_agent_api_failure(self, client, valid_jwt_token, mock_mongodb, mock_chat_handler):
        """Test chat endpoint when agent API fails"""
        mock_chat_handler.call_agent_api.return_value = {
            "success": False,
            "error": "API timeout",
            "response": "Sorry, I'm having trouble right now."
        }
        
        with patch('api.endpoints.mongodb_client', mock_mongodb), \
             patch('api.endpoints.chat_handler', mock_chat_handler):
            
            response = client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["response"] == "Sorry, I'm having trouble right now."
            assert "error" in data["metadata"]

class TestSaveProgressEndpoint:
    
    def test_save_progress_success(self, client, valid_jwt_token, mock_mongodb):
        """Test successful progress saving"""
        with patch('api.endpoints.mongodb_client', mock_mongodb):
            
            response = client.post(
                "/api/v1/save_progress",
                json={
                    "chat_history": [
                        {
                            "role": "user",
                            "content": "Hello",
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        {
                            "role": "assistant",
                            "content": "Hi there!",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    ],
                    "session_metadata": {"session_type": "test"}
                },
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["user_id"] == "test-user-123"
            assert data["saved_messages"] == 2
            assert "timestamp" in data
    
    def test_save_progress_unauthorized(self, client):
        """Test save progress endpoint without authorization"""
        response = client.post(
            "/api/v1/save_progress",
            json={"chat_history": []}
        )
        
        assert response.status_code == 401

class TestChatHistoryEndpoint:
    
    def test_get_chat_history_success(self, client, valid_jwt_token, mock_mongodb, sample_chat_history):
        """Test successful chat history retrieval"""
        mock_mongodb.get_user_chat_history.return_value = sample_chat_history
        
        with patch('api.endpoints.mongodb_client', mock_mongodb):
            
            response = client.get(
                "/api/v1/chat_history?limit=10&skip=0",
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test-user-123"
            assert len(data["chat_history"]) == 2
            assert data["count"] == 2
            assert data["limit"] == 10
            assert data["skip"] == 0
    
    def test_get_chat_history_unauthorized(self, client):
        """Test chat history endpoint without authorization"""
        response = client.get("/api/v1/chat_history")
        
        assert response.status_code == 401

class TestUserSessionEndpoint:
    
    def test_get_user_session_success(self, client, valid_jwt_token, mock_mongodb):
        """Test successful user session retrieval"""
        mock_session = {
            "_id": "session123",
            "user_id": "test-user-123",
            "chat_history": [],
            "last_seen": datetime.utcnow(),
            "session_metadata": {}
        }
        mock_mongodb.get_user_session.return_value = mock_session
        
        with patch('api.endpoints.mongodb_client', mock_mongodb):
            
            response = client.get(
                "/api/v1/user_session",
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test-user-123"
            assert data["session"] is not None
    
    def test_get_user_session_not_found(self, client, valid_jwt_token, mock_mongodb):
        """Test user session retrieval when no session exists"""
        mock_mongodb.get_user_session.return_value = None
        
        with patch('api.endpoints.mongodb_client', mock_mongodb):
            
            response = client.get(
                "/api/v1/user_session",
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test-user-123"
            assert data["session"] is None
            assert "no active session" in data["message"].lower()

class TestDeleteUserDataEndpoint:
    
    def test_delete_user_data_success(self, client, valid_jwt_token, mock_mongodb):
        """Test successful user data deletion"""
        mock_mongodb.delete_user_data.return_value = {
            "chat_messages": 5,
            "sessions": 1
        }
        
        with patch('api.endpoints.mongodb_client', mock_mongodb):
            
            response = client.delete(
                "/api/v1/user_data",
                headers={"Authorization": f"Bearer {valid_jwt_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test-user-123"
            assert data["deleted"]["chat_messages"] == 5
            assert data["deleted"]["sessions"] == 1
            assert "deleted" in data["message"].lower()
