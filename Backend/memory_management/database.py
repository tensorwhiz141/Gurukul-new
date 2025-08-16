"""
Database integration for the Memory Management System.

This module handles all database operations for memory storage and retrieval,
integrating with the existing MongoDB infrastructure.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from .models import (
    MemoryCreateRequest, InteractionCreateRequest, MemoryUpdateRequest,
    MemoryChunkResponse, InteractionResponse, PersonaMemorySummary,
    ContentType, ImportanceLevel
)

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MEMORY_DB_NAME", "gurukul")
MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "365"))

# Collection names
MEMORY_CHUNKS_COLLECTION = "memory_chunks"
MEMORY_INTERACTIONS_COLLECTION = "memory_interactions"
PERSONA_MEMORY_INDEX_COLLECTION = "persona_memory_index"


class MemoryDatabase:
    """Database manager for memory operations."""
    
    def __init__(self):
        """Initialize database connection and collections."""
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._memory_chunks: Optional[Collection] = None
        self._memory_interactions: Optional[Collection] = None
        self._persona_index: Optional[Collection] = None
        self._initialize_connection()
        self._ensure_indexes()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection."""
        try:
            self._client = MongoClient(MONGODB_URL)
            self._db = self._client[DB_NAME]
            
            # Test connection
            self._client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {DB_NAME}")
            
            # Initialize collections
            self._memory_chunks = self._db[MEMORY_CHUNKS_COLLECTION]
            self._memory_interactions = self._db[MEMORY_INTERACTIONS_COLLECTION]
            self._persona_index = self._db[PERSONA_MEMORY_INDEX_COLLECTION]
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _ensure_indexes(self):
        """Create necessary database indexes for optimal performance."""
        try:
            # Memory chunks indexes
            self._memory_chunks.create_index([
                ("user_id", ASCENDING),
                ("persona_id", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="user_persona_timestamp")
            
            self._memory_chunks.create_index([
                ("persona_id", ASCENDING),
                ("is_active", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="persona_active_timestamp")
            
            self._memory_chunks.create_index([
                ("content_type", ASCENDING),
                ("metadata.importance", DESCENDING)
            ], name="content_type_importance")
            
            self._memory_chunks.create_index([
                ("metadata.tags", ASCENDING)
            ], name="metadata_tags")
            
            # Text index for content search
            self._memory_chunks.create_index([
                ("content", TEXT),
                ("metadata.topic", TEXT)
            ], name="content_search")
            
            # TTL index for automatic cleanup
            self._memory_chunks.create_index([
                ("created_at", ASCENDING)
            ], expireAfterSeconds=MEMORY_RETENTION_DAYS * 24 * 3600, name="ttl_cleanup")
            
            # Memory interactions indexes
            self._memory_interactions.create_index([
                ("user_id", ASCENDING),
                ("persona_id", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="user_persona_timestamp")
            
            self._memory_interactions.create_index([
                ("persona_id", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="persona_timestamp")
            
            # Persona index
            self._persona_index.create_index([
                ("persona_id", ASCENDING),
                ("user_id", ASCENDING)
            ], unique=True, name="persona_user_unique")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            # Don't raise here as the application can still function without optimal indexes
    
    def create_memory_chunk(self, request: MemoryCreateRequest) -> str:
        """
        Create a new memory chunk.
        
        Args:
            request: Memory creation request
            
        Returns:
            str: Created memory ID
            
        Raises:
            PyMongoError: If database operation fails
        """
        memory_id = str(uuid4())
        now = datetime.utcnow()
        
        document = {
            "memory_id": memory_id,
            "user_id": request.user_id,
            "persona_id": request.persona_id,
            "content": request.content,
            "content_type": request.content_type.value,
            "metadata": {
                "tags": request.metadata.tags,
                "importance": request.metadata.importance.value,
                "topic": request.metadata.topic,
                "context_type": request.metadata.context_type,
                "source": request.metadata.source,
                "confidence": request.metadata.confidence,
                "related_memories": request.metadata.related_memories
            },
            "timestamp": request.timestamp,
            "created_at": now,
            "updated_at": now,
            "is_active": True
        }
        
        try:
            result = self._memory_chunks.insert_one(document)
            logger.info(f"Created memory chunk: {memory_id} for persona: {request.persona_id}")
            
            # Update persona index
            self._update_persona_index(request.persona_id, request.user_id)
            
            return memory_id
            
        except PyMongoError as e:
            logger.error(f"Failed to create memory chunk: {e}")
            raise
    
    def create_interaction(self, request: InteractionCreateRequest) -> str:
        """
        Create a new interaction record.
        
        Args:
            request: Interaction creation request
            
        Returns:
            str: Created interaction ID
            
        Raises:
            PyMongoError: If database operation fails
        """
        interaction_id = str(uuid4())
        now = datetime.utcnow()
        
        document = {
            "interaction_id": interaction_id,
            "user_id": request.user_id,
            "persona_id": request.persona_id,
            "user_message": request.user_message,
            "agent_response": request.agent_response,
            "context": {
                "session_id": request.context.session_id,
                "conversation_turn": request.context.conversation_turn,
                "domain": request.context.domain,
                "intent": request.context.intent,
                "previous_context": request.context.previous_context
            },
            "metadata": {
                "response_time": request.metadata.response_time,
                "confidence": request.metadata.confidence,
                "feedback": request.metadata.feedback,
                "tags": request.metadata.tags,
                "model_used": request.metadata.model_used
            },
            "timestamp": request.timestamp,
            "created_at": now,
            "is_active": True
        }
        
        try:
            result = self._memory_interactions.insert_one(document)
            logger.info(f"Created interaction: {interaction_id} for persona: {request.persona_id}")
            
            # Update persona index
            self._update_persona_index(request.persona_id, request.user_id)
            
            return interaction_id
            
        except PyMongoError as e:
            logger.error(f"Failed to create interaction: {e}")
            raise
    
    def get_memories_by_persona(
        self, 
        persona_id: str, 
        user_id: Optional[str] = None,
        limit: int = 50, 
        offset: int = 0,
        content_types: Optional[List[ContentType]] = None,
        min_importance: Optional[ImportanceLevel] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieve memories for a specific persona.
        
        Args:
            persona_id: Persona identifier
            user_id: Optional user filter
            limit: Maximum number of results
            offset: Number of results to skip
            content_types: Filter by content types
            min_importance: Minimum importance level
            
        Returns:
            Tuple of (memories list, total count)
        """
        query = {
            "persona_id": persona_id,
            "is_active": True
        }
        
        if user_id:
            query["user_id"] = user_id
        
        if content_types:
            query["content_type"] = {"$in": [ct.value for ct in content_types]}
        
        if min_importance:
            query["metadata.importance"] = {"$gte": min_importance.value}
        
        try:
            # Get total count
            total_count = self._memory_chunks.count_documents(query)
            
            # Get paginated results
            cursor = self._memory_chunks.find(query).sort([
                ("metadata.importance", DESCENDING),
                ("timestamp", DESCENDING)
            ]).skip(offset).limit(limit)
            
            memories = list(cursor)
            
            # Convert ObjectId to string
            for memory in memories:
                memory["_id"] = str(memory["_id"])
            
            logger.info(f"Retrieved {len(memories)} memories for persona: {persona_id}")
            return memories, total_count
            
        except PyMongoError as e:
            logger.error(f"Failed to retrieve memories for persona {persona_id}: {e}")
            raise
    
    def get_recent_interactions(
        self, 
        limit: int = 20, 
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent interactions for chain-of-thought processing.
        
        Args:
            limit: Maximum number of interactions
            user_id: Optional user filter
            persona_id: Optional persona filter
            
        Returns:
            List of recent interactions
        """
        query = {"is_active": True}
        
        if user_id:
            query["user_id"] = user_id
        
        if persona_id:
            query["persona_id"] = persona_id
        
        try:
            cursor = self._memory_interactions.find(query).sort([
                ("timestamp", DESCENDING)
            ]).limit(limit)
            
            interactions = list(cursor)
            
            # Convert ObjectId to string and reverse for chronological order
            for interaction in interactions:
                interaction["_id"] = str(interaction["_id"])
            
            interactions.reverse()  # Most recent last for chain-of-thought
            
            logger.info(f"Retrieved {len(interactions)} recent interactions")
            return interactions
            
        except PyMongoError as e:
            logger.error(f"Failed to retrieve recent interactions: {e}")
            raise
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory data or None if not found
        """
        try:
            memory = self._memory_chunks.find_one({"memory_id": memory_id, "is_active": True})
            if memory:
                memory["_id"] = str(memory["_id"])
            return memory

        except PyMongoError as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise

    def get_persona_memory_summary(self, persona_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get memory summary for a persona.

        Args:
            persona_id: Persona identifier
            user_id: Optional user filter

        Returns:
            Persona memory summary or None if not found
        """
        try:
            query = {"persona_id": persona_id}
            if user_id:
                query["user_id"] = user_id

            summary = self._persona_index.find_one(query)
            if summary:
                summary["_id"] = str(summary["_id"])

                # Get recent topics
                recent_memories = self._memory_chunks.find(
                    {**query, "is_active": True, "metadata.topic": {"$ne": None}},
                    {"metadata.topic": 1}
                ).sort("timestamp", DESCENDING).limit(10)

                topics = list(set(mem.get("metadata", {}).get("topic") for mem in recent_memories if mem.get("metadata", {}).get("topic")))
                summary["recent_topics"] = topics[:5]

                # Get importance distribution
                importance_pipeline = [
                    {"$match": {**query, "is_active": True}},
                    {"$group": {"_id": "$metadata.importance", "count": {"$sum": 1}}}
                ]

                importance_dist = {}
                for result in self._memory_chunks.aggregate(importance_pipeline):
                    importance_dist[str(result["_id"])] = result["count"]

                summary["importance_distribution"] = importance_dist

            return summary

        except PyMongoError as e:
            logger.error(f"Failed to get persona summary {persona_id}: {e}")
            raise

    def search_memories(
        self,
        query: str,
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        content_types: Optional[List[ContentType]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search memories using text query.

        Args:
            query: Search query
            persona_id: Optional persona filter
            user_id: Optional user filter
            content_types: Optional content type filter
            limit: Maximum results

        Returns:
            List of matching memories
        """
        try:
            search_query = {
                "$text": {"$search": query},
                "is_active": True
            }

            if persona_id:
                search_query["persona_id"] = persona_id

            if user_id:
                search_query["user_id"] = user_id

            if content_types:
                search_query["content_type"] = {"$in": [ct.value for ct in content_types]}

            cursor = self._memory_chunks.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            results = list(cursor)
            for result in results:
                result["_id"] = str(result["_id"])

            return results

        except PyMongoError as e:
            logger.error(f"Failed to search memories: {e}")
            raise

    def update_memory_chunk(self, memory_id: str, request: MemoryUpdateRequest) -> bool:
        """
        Update an existing memory chunk.

        Args:
            memory_id: Memory identifier
            request: Update request

        Returns:
            bool: True if updated successfully
        """
        try:
            update_doc = {"$set": {"updated_at": datetime.utcnow()}}

            if request.content is not None:
                update_doc["$set"]["content"] = request.content

            if request.content_type is not None:
                update_doc["$set"]["content_type"] = request.content_type.value

            if request.metadata is not None:
                update_doc["$set"]["metadata"] = {
                    "tags": request.metadata.tags,
                    "importance": request.metadata.importance.value,
                    "topic": request.metadata.topic,
                    "context_type": request.metadata.context_type,
                    "source": request.metadata.source,
                    "confidence": request.metadata.confidence,
                    "related_memories": request.metadata.related_memories
                }

            if request.is_active is not None:
                update_doc["$set"]["is_active"] = request.is_active

            result = self._memory_chunks.update_one(
                {"memory_id": memory_id, "is_active": True},
                update_doc
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise

    def delete_memory_chunk(self, memory_id: str, hard_delete: bool = False) -> bool:
        """
        Delete a memory chunk.

        Args:
            memory_id: Memory identifier
            hard_delete: If True, permanently delete; if False, soft delete

        Returns:
            bool: True if deleted successfully
        """
        try:
            if hard_delete:
                result = self._memory_chunks.delete_one({"memory_id": memory_id})
                return result.deleted_count > 0
            else:
                result = self._memory_chunks.update_one(
                    {"memory_id": memory_id, "is_active": True},
                    {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
                )
                return result.modified_count > 0

        except PyMongoError as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    def _update_persona_index(self, persona_id: str, user_id: str):
        """Update the persona memory index with latest statistics."""
        try:
            # Count memories by content type
            pipeline = [
                {"$match": {"persona_id": persona_id, "user_id": user_id, "is_active": True}},
                {"$group": {
                    "_id": "$content_type",
                    "count": {"$sum": 1}
                }}
            ]

            content_counts = {}
            for result in self._memory_chunks.aggregate(pipeline):
                content_counts[result["_id"]] = result["count"]

            # Get last interaction timestamp
            last_interaction = self._memory_interactions.find_one(
                {"persona_id": persona_id, "user_id": user_id, "is_active": True},
                sort=[("timestamp", DESCENDING)]
            )

            last_interaction_time = last_interaction["timestamp"] if last_interaction else None

            # Update or create persona index
            update_doc = {
                "$set": {
                    "persona_id": persona_id,
                    "user_id": user_id,
                    "total_memories": sum(content_counts.values()),
                    "memory_categories": content_counts,
                    "last_interaction": last_interaction_time,
                    "updated_at": datetime.utcnow()
                }
            }

            self._persona_index.update_one(
                {"persona_id": persona_id, "user_id": user_id},
                update_doc,
                upsert=True
            )

        except Exception as e:
            logger.error(f"Failed to update persona index: {e}")
            # Don't raise as this is not critical for main functionality
    
    def close_connection(self):
        """Close database connection."""
        if self._client:
            self._client.close()
            logger.info("Database connection closed")


# Global database instance
_db_instance: Optional[MemoryDatabase] = None


def get_memory_database() -> MemoryDatabase:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = MemoryDatabase()
    return _db_instance
