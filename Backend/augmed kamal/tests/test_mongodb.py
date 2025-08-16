"""
Tests for MongoDB client
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from memory.mongodb_client import MongoDBClient

class TestMongoDBClient:
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful MongoDB connection"""
        with patch('memory.mongodb_client.MongoClient') as mock_mongo:
            mock_client = Mock()
            mock_client.admin.command = Mock()
            mock_db = Mock()
            mock_client.__getitem__ = Mock(return_value=mock_db)
            mock_mongo.return_value = mock_client

            mongodb_client = MongoDBClient()
            await mongodb_client.connect()

            assert mongodb_client.client is not None
            assert mongodb_client.db is not None
            mock_client.admin.command.assert_called_with('ping')
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when MongoDB is healthy"""
        mongodb_client = MongoDBClient()
        
        with patch.object(mongodb_client, 'client') as mock_client:
            mock_client.admin.command = Mock()
            
            result = await mongodb_client.health_check()
            
            assert result["status"] == "healthy"
            assert "database" in result
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when MongoDB is unhealthy"""
        mongodb_client = MongoDBClient()
        mongodb_client.client = None
        
        result = await mongodb_client.health_check()
        
        assert result["status"] == "disconnected"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_save_chat_message(self):
        """Test saving a chat message"""
        mongodb_client = MongoDBClient()
        
        mock_collection = Mock()
        mock_collection.insert_one.return_value.inserted_id = "test-id"
        mongodb_client.chat_history_collection = mock_collection
        
        result = await mongodb_client.save_chat_message(
            user_id="test-user",
            message="Hello",
            response="Hi there!",
            metadata={"test": "data"}
        )
        
        assert result == "test-id"
        mock_collection.insert_one.assert_called_once()
        
        # Check the document structure
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["user_id"] == "test-user"
        assert call_args["message"] == "Hello"
        assert call_args["response"] == "Hi there!"
        assert call_args["metadata"]["test"] == "data"
        assert "timestamp" in call_args
    
    @pytest.mark.asyncio
    async def test_get_user_chat_history(self):
        """Test retrieving user chat history"""
        mongodb_client = MongoDBClient()
        
        mock_docs = [
            {
                "_id": "id1",
                "user_id": "test-user",
                "message": "Hello",
                "response": "Hi",
                "timestamp": datetime.utcnow()
            },
            {
                "_id": "id2",
                "user_id": "test-user",
                "message": "How are you?",
                "response": "Good",
                "timestamp": datetime.utcnow()
            }
        ]
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_docs))
        
        mock_collection = Mock()
        mock_collection.find.return_value.sort.return_value.skip.return_value.limit.return_value = mock_cursor
        mongodb_client.chat_history_collection = mock_collection
        
        result = await mongodb_client.get_user_chat_history("test-user", limit=10, skip=0)
        
        assert len(result) == 2
        assert result[0]["_id"] == "id1"
        assert result[1]["_id"] == "id2"
        
        # Verify the query
        mock_collection.find.assert_called_with({"user_id": "test-user"})
    
    @pytest.mark.asyncio
    async def test_save_user_session(self):
        """Test saving user session"""
        mongodb_client = MongoDBClient()
        
        mock_collection = Mock()
        mock_result = Mock()
        mock_result.upserted_id = "session-id"
        mock_collection.replace_one.return_value = mock_result
        mongodb_client.user_sessions_collection = mock_collection
        
        chat_history = [{"role": "user", "content": "Hello"}]
        result = await mongodb_client.save_user_session(
            user_id="test-user",
            chat_history=chat_history,
            session_metadata={"test": "data"}
        )
        
        assert result == "session-id"
        mock_collection.replace_one.assert_called_once()
        
        # Check the call arguments
        call_args = mock_collection.replace_one.call_args
        filter_doc = call_args[0][0]
        session_doc = call_args[0][1]
        
        assert filter_doc["user_id"] == "test-user"
        assert session_doc["user_id"] == "test-user"
        assert session_doc["chat_history"] == chat_history
        assert session_doc["session_metadata"]["test"] == "data"
        assert "last_seen" in session_doc
    
    @pytest.mark.asyncio
    async def test_get_user_session(self):
        """Test retrieving user session"""
        mongodb_client = MongoDBClient()
        
        mock_session = {
            "_id": "session-id",
            "user_id": "test-user",
            "chat_history": [],
            "last_seen": datetime.utcnow()
        }
        
        mock_collection = Mock()
        mock_collection.find_one.return_value = mock_session
        mongodb_client.user_sessions_collection = mock_collection
        
        result = await mongodb_client.get_user_session("test-user")
        
        assert result["_id"] == "session-id"
        assert result["user_id"] == "test-user"
        mock_collection.find_one.assert_called_with({"user_id": "test-user"})
    
    @pytest.mark.asyncio
    async def test_get_user_session_not_found(self):
        """Test retrieving user session when none exists"""
        mongodb_client = MongoDBClient()
        
        mock_collection = Mock()
        mock_collection.find_one.return_value = None
        mongodb_client.user_sessions_collection = mock_collection
        
        result = await mongodb_client.get_user_session("test-user")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_user_data(self):
        """Test deleting all user data"""
        mongodb_client = MongoDBClient()
        
        mock_chat_result = Mock()
        mock_chat_result.deleted_count = 5
        mock_session_result = Mock()
        mock_session_result.deleted_count = 1
        
        mock_chat_collection = Mock()
        mock_chat_collection.delete_many.return_value = mock_chat_result
        mock_session_collection = Mock()
        mock_session_collection.delete_many.return_value = mock_session_result
        
        mongodb_client.chat_history_collection = mock_chat_collection
        mongodb_client.user_sessions_collection = mock_session_collection
        
        result = await mongodb_client.delete_user_data("test-user")
        
        assert result["chat_messages"] == 5
        assert result["sessions"] == 1
        
        mock_chat_collection.delete_many.assert_called_with({"user_id": "test-user"})
        mock_session_collection.delete_many.assert_called_with({"user_id": "test-user"})
