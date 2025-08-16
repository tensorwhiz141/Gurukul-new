"""
MongoDB Client for Agent Memory System
Handles user chat history and session storage
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import logging

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = os.getenv("MONGODB_DATABASE", "agent_memory")
        self.client = None
        self.db = None
        self.chat_history_collection = None
        self.user_sessions_collection = None
    
    async def connect(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(self.uri)
            # Test the connection
            self.client.admin.command('ping')

            self.db = self.client[self.database_name]
            self.chat_history_collection = self.db.chat_history
            self.user_sessions_collection = self.db.user_sessions

            # Create indexes for better performance
            self.chat_history_collection.create_index("user_id")
            self.chat_history_collection.create_index("timestamp")
            self.user_sessions_collection.create_index("user_id")

            logger.info("MongoDB connection established successfully")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            # Don't raise - allow app to continue without MongoDB
            self.client = None
        except Exception as e:
            logger.error(f"MongoDB initialization error: {str(e)}")
            # Don't raise - allow app to continue without MongoDB
            self.client = None
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB connection health"""
        try:
            if not self.client:
                return {"status": "disconnected", "error": "No client connection"}
            
            # Ping the database
            self.client.admin.command('ping')
            return {"status": "healthy", "database": self.database_name}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def save_chat_message(
        self,
        user_id: str,
        message: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a single chat exchange to history

        Args:
            user_id: User identifier from Supabase
            message: User's message
            response: Agent's response
            metadata: Additional metadata (e.g., model used, tokens, etc.)

        Returns:
            Document ID of the saved chat message
        """
        if not self.client or not self.chat_history_collection:
            logger.warning("MongoDB not connected - chat message not saved")
            return "no-mongodb-connection"

        try:
            chat_document = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }

            result = self.chat_history_collection.insert_one(chat_document)
            logger.info(f"Chat message saved for user {user_id}")
            return str(result.inserted_id)

        except PyMongoError as e:
            logger.error(f"Failed to save chat message: {str(e)}")
            raise
    
    async def get_user_chat_history(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve user's chat history

        Args:
            user_id: User identifier
            limit: Maximum number of messages to retrieve
            skip: Number of messages to skip (for pagination)

        Returns:
            List of chat messages sorted by timestamp (newest first)
        """
        if not self.client or not self.chat_history_collection:
            logger.warning("MongoDB not connected - returning empty chat history")
            return []

        try:
            cursor = self.chat_history_collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).skip(skip).limit(limit)

            history = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                history.append(doc)

            logger.info(f"Retrieved {len(history)} chat messages for user {user_id}")
            return history

        except PyMongoError as e:
            logger.error(f"Failed to retrieve chat history: {str(e)}")
            raise
    
    async def save_user_session(
        self, 
        user_id: str, 
        chat_history: List[Dict[str, Any]],
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save or update user's current session
        
        Args:
            user_id: User identifier
            chat_history: Current session's chat messages
            session_metadata: Additional session information
            
        Returns:
            Document ID of the saved session
        """
        try:
            session_document = {
                "user_id": user_id,
                "chat_history": chat_history,
                "last_seen": datetime.utcnow(),
                "session_metadata": session_metadata or {}
            }
            
            # Upsert: update if exists, insert if not
            result = self.user_sessions_collection.replace_one(
                {"user_id": user_id},
                session_document,
                upsert=True
            )
            
            logger.info(f"Session saved for user {user_id}")
            return str(result.upserted_id) if result.upserted_id else "updated"
            
        except PyMongoError as e:
            logger.error(f"Failed to save user session: {str(e)}")
            raise
    
    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user's current session
        
        Args:
            user_id: User identifier
            
        Returns:
            User session document or None if not found
        """
        try:
            session = self.user_sessions_collection.find_one({"user_id": user_id})
            
            if session:
                session["_id"] = str(session["_id"])
                logger.info(f"Retrieved session for user {user_id}")
            else:
                logger.info(f"No session found for user {user_id}")
            
            return session
            
        except PyMongoError as e:
            logger.error(f"Failed to retrieve user session: {str(e)}")
            raise
    
    async def delete_user_data(self, user_id: str) -> Dict[str, int]:
        """
        Delete all data for a user (GDPR compliance)
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with count of deleted documents
        """
        try:
            chat_result = self.chat_history_collection.delete_many({"user_id": user_id})
            session_result = self.user_sessions_collection.delete_many({"user_id": user_id})
            
            deleted_counts = {
                "chat_messages": chat_result.deleted_count,
                "sessions": session_result.deleted_count
            }
            
            logger.info(f"Deleted user data for {user_id}: {deleted_counts}")
            return deleted_counts
            
        except PyMongoError as e:
            logger.error(f"Failed to delete user data: {str(e)}")
            raise
