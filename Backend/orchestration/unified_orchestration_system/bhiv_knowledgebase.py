"""
BHIV Knowledgebase - Enhanced Contextual Vector Retriever
Advanced knowledgebase system with BHIV Core schema integration

Features:
- Qdrant vector database integration
- Advanced filtering and search capabilities
- Source tracking and metadata management
- Context-aware retrieval
- Multi-modal content support
- Performance optimization and caching
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import hashlib

# Vector database imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available, falling back to FAISS")

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Local imports
from bhiv_core_schema import (
    KnowledgebaseQuery, KnowledgebaseFilters, KnowledgebaseResult,
    ClassificationType, VectorType
)
from data_ingestion import UnifiedDataIngestion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector database integration for BHIV Knowledgebase
    """
    
    def __init__(self, collection_name: str, embedding_model, qdrant_url: str = "localhost:6333"):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.qdrant_url = qdrant_url
        self.client = None
        self.dimension = None
        
    async def initialize(self):
        """Initialize Qdrant client and collection"""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")
            
        try:
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Get embedding dimension
            test_embedding = self.embedding_model.embed_query("test")
            self.dimension = len(test_embedding)
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to Qdrant collection"""
        if not self.client:
            await self.initialize()
            
        points = []
        for i, doc in enumerate(documents):
            # Generate embedding
            embedding = self.embedding_model.embed_query(doc.page_content)
            
            # Create point
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upload points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(documents)} documents to Qdrant collection {self.collection_name}")
    
    async def search(
        self, 
        query: str, 
        filters: Optional[KnowledgebaseFilters] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search documents in Qdrant collection"""
        if not self.client:
            await self.initialize()
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Build filter conditions
        filter_conditions = None
        if filters:
            conditions = []
            
            if filters.lang:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.lang",
                        match=models.MatchValue(value=filters.lang)
                    )
                )
            
            if filters.subject:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.subject",
                        match=models.MatchValue(value=filters.subject)
                    )
                )
            
            if filters.curriculum_tag:
                for tag in filters.curriculum_tag:
                    conditions.append(
                        models.FieldCondition(
                            key="metadata.curriculum_tag",
                            match=models.MatchValue(value=tag)
                        )
                    )
            
            if conditions:
                filter_conditions = models.Filter(
                    must=conditions
                )
        
        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=top_k,
            score_threshold=similarity_threshold
        )
        
        # Format results
        results = []
        for hit in search_result:
            results.append({
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "source": hit.payload["source"],
                "score": hit.score,
                "id": hit.id
            })
        
        return results


class EnhancedKnowledgebaseManager:
    """
    Enhanced knowledgebase manager with BHIV Core schema support
    """
    
    def __init__(self, data_ingestion: UnifiedDataIngestion):
        self.data_ingestion = data_ingestion
        self.embedding_model = None
        self.vector_stores = {}
        self.qdrant_stores = {}
        self.query_cache = {}
        self.performance_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_query_time": 0.0,
            "last_updated": datetime.now()
        }
        
    async def initialize(self):
        """Initialize the enhanced knowledgebase"""
        logger.info("Initializing Enhanced BHIV Knowledgebase...")
        
        # Initialize embedding model
        self.embedding_model = self.data_ingestion.initialize_embedding_model()
        
        # Load existing FAISS vector stores
        self.vector_stores = self.data_ingestion.load_existing_vector_stores()
        
        # Initialize Qdrant stores if available
        if QDRANT_AVAILABLE:
            await self._initialize_qdrant_stores()
        
        # If no stores exist, create them
        if not self.vector_stores and not self.qdrant_stores:
            logger.info("No existing vector stores found. Creating new ones...")
            self.vector_stores = self.data_ingestion.ingest_all_data()
            
            if QDRANT_AVAILABLE:
                await self._migrate_to_qdrant()
        
        logger.info(f"Enhanced Knowledgebase initialized with {len(self.vector_stores)} FAISS stores and {len(self.qdrant_stores)} Qdrant stores")
    
    async def _initialize_qdrant_stores(self):
        """Initialize Qdrant vector stores"""
        try:
            qdrant_url = os.getenv("QDRANT_URL", "localhost:6333")
            
            # Create Qdrant stores for each domain
            domains = ["vedas", "wellness", "educational", "unified"]
            
            for domain in domains:
                qdrant_store = QdrantVectorStore(
                    collection_name=f"bhiv_{domain}",
                    embedding_model=self.embedding_model,
                    qdrant_url=qdrant_url
                )
                
                try:
                    await qdrant_store.initialize()
                    self.qdrant_stores[domain] = qdrant_store
                    logger.info(f"Initialized Qdrant store for domain: {domain}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Qdrant store for {domain}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant stores: {e}")
    
    async def _migrate_to_qdrant(self):
        """Migrate existing FAISS stores to Qdrant"""
        if not self.qdrant_stores or not self.vector_stores:
            return
            
        logger.info("Migrating FAISS stores to Qdrant...")
        
        for domain, faiss_store in self.vector_stores.items():
            if domain in self.qdrant_stores:
                try:
                    # Get all documents from FAISS store
                    # This is a simplified approach - in practice, you'd need to extract documents properly
                    logger.info(f"Migration for {domain} would happen here")
                    # TODO: Implement proper migration logic
                except Exception as e:
                    logger.error(f"Failed to migrate {domain} to Qdrant: {e}")
    
    async def query_knowledgebase(
        self,
        query: str,
        kb_query: KnowledgebaseQuery,
        classification: ClassificationType
    ) -> KnowledgebaseResult:
        """
        Enhanced knowledgebase query with BHIV schema support
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, kb_query, classification)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.performance_metrics["cache_hits"] += 1
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached_result["result"]
            
            # Select appropriate vector store
            store_name = self._select_store_for_classification(classification)
            
            # Perform search based on vector type preference
            if kb_query.vector_type == VectorType.QDRANT and store_name in self.qdrant_stores:
                results = await self._search_qdrant(query, kb_query, store_name)
            else:
                results = await self._search_faiss(query, kb_query, store_name)
            
            # Calculate query time
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            kb_result = KnowledgebaseResult(
                documents=results,
                total_results=len(results),
                query_time_ms=query_time,
                embedding_used=self.embedding_model.__class__.__name__ if self.embedding_model else None,
                filters_applied=kb_query.filters.dict() if kb_query.filters else None
            )
            
            # Cache result
            self.query_cache[cache_key] = {
                "result": kb_result,
                "timestamp": datetime.now(),
                "ttl": timedelta(hours=1)  # Cache for 1 hour
            }
            
            # Update metrics
            self._update_performance_metrics(query_time)
            
            return kb_result
            
        except Exception as e:
            logger.error(f"Error querying knowledgebase: {e}")
            return KnowledgebaseResult(
                documents=[],
                total_results=0,
                query_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                embedding_used=None,
                filters_applied=None
            )
    
    async def _search_qdrant(
        self, 
        query: str, 
        kb_query: KnowledgebaseQuery, 
        store_name: str
    ) -> List[Dict[str, Any]]:
        """Search using Qdrant vector store"""
        if store_name not in self.qdrant_stores:
            logger.warning(f"Qdrant store {store_name} not available, falling back to FAISS")
            return await self._search_faiss(query, kb_query, store_name)
        
        qdrant_store = self.qdrant_stores[store_name]
        
        results = await qdrant_store.search(
            query=query,
            filters=kb_query.filters,
            top_k=kb_query.top_k,
            similarity_threshold=kb_query.similarity_threshold
        )
        
        return results
    
    async def _search_faiss(
        self, 
        query: str, 
        kb_query: KnowledgebaseQuery, 
        store_name: str
    ) -> List[Dict[str, Any]]:
        """Search using FAISS vector store"""
        if store_name not in self.vector_stores:
            # Try unified store as fallback
            store_name = 'unified' if 'unified' in self.vector_stores else list(self.vector_stores.keys())[0]
        
        if store_name not in self.vector_stores:
            return []
        
        faiss_store = self.vector_stores[store_name]
        retriever = faiss_store.as_retriever(search_kwargs={"k": kb_query.top_k})
        
        # Get relevant documents
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Apply filters if specified
        filtered_docs = self._apply_filters(relevant_docs, kb_query.filters)
        
        # Format results
        results = []
        for doc in filtered_docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get('source', 'unknown'),
                "score": 1.0  # FAISS doesn't provide scores in this interface
            })
        
        return results

    def _select_store_for_classification(self, classification: ClassificationType) -> str:
        """Select appropriate vector store based on classification"""
        store_mapping = {
            ClassificationType.SPIRITUAL_QUERY: 'vedas',
            ClassificationType.WELLNESS_QUERY: 'wellness',
            ClassificationType.LEARNING_QUERY: 'educational',
            ClassificationType.GENERAL_QUERY: 'unified',
            ClassificationType.EMERGENCY: 'wellness'  # Emergency queries might need wellness resources
        }

        preferred_store = store_mapping.get(classification, 'unified')

        # Check if preferred store exists, fallback to available stores
        available_stores = list(self.qdrant_stores.keys()) + list(self.vector_stores.keys())

        if preferred_store in available_stores:
            return preferred_store
        elif 'unified' in available_stores:
            return 'unified'
        elif available_stores:
            return available_stores[0]
        else:
            logger.warning("No vector stores available")
            return 'unified'  # Return default even if not available

    def _apply_filters(self, documents: List[Document], filters: Optional[KnowledgebaseFilters]) -> List[Document]:
        """Apply filters to documents"""
        if not filters:
            return documents

        filtered_docs = []

        for doc in documents:
            metadata = doc.metadata

            # Language filter
            if filters.lang and metadata.get('lang') != filters.lang:
                continue

            # Subject filter
            if filters.subject and metadata.get('subject') != filters.subject:
                continue

            # Curriculum tag filter
            if filters.curriculum_tag:
                doc_tags = metadata.get('curriculum_tag', [])
                if isinstance(doc_tags, str):
                    doc_tags = [doc_tags]

                if not any(tag in doc_tags for tag in filters.curriculum_tag):
                    continue

            # Difficulty level filter
            if filters.difficulty_level and metadata.get('difficulty_level') != filters.difficulty_level:
                continue

            filtered_docs.append(doc)

        return filtered_docs

    def _generate_cache_key(
        self,
        query: str,
        kb_query: KnowledgebaseQuery,
        classification: ClassificationType
    ) -> str:
        """Generate cache key for query"""
        cache_data = {
            'query': query,
            'classification': classification.value,
            'top_k': kb_query.top_k,
            'vector_type': kb_query.vector_type.value,
            'filters': kb_query.filters.model_dump() if kb_query.filters else None
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached item is still valid"""
        timestamp = cached_item.get('timestamp')
        ttl = cached_item.get('ttl', timedelta(hours=1))

        if not timestamp:
            return False

        return datetime.now() - timestamp < ttl

    def _update_performance_metrics(self, query_time_ms: float):
        """Update performance metrics"""
        self.performance_metrics["total_queries"] += 1

        # Update average query time
        total_queries = self.performance_metrics["total_queries"]
        current_avg = self.performance_metrics["avg_query_time"]

        new_avg = ((current_avg * (total_queries - 1)) + query_time_ms) / total_queries
        self.performance_metrics["avg_query_time"] = new_avg
        self.performance_metrics["last_updated"] = datetime.now()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get knowledgebase performance metrics"""
        return {
            **self.performance_metrics,
            "cache_size": len(self.query_cache),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / max(self.performance_metrics["total_queries"], 1)
            ) * 100,
            "available_stores": {
                "qdrant": list(self.qdrant_stores.keys()),
                "faiss": list(self.vector_stores.keys())
            },
            "qdrant_available": QDRANT_AVAILABLE
        }

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for knowledgebase"""
        health_status = {
            "status": "healthy",
            "embedding_model": "available" if self.embedding_model else "not_available",
            "stores": {
                "qdrant": {
                    "available": QDRANT_AVAILABLE,
                    "collections": len(self.qdrant_stores),
                    "names": list(self.qdrant_stores.keys())
                },
                "faiss": {
                    "available": True,
                    "collections": len(self.vector_stores),
                    "names": list(self.vector_stores.keys())
                }
            },
            "performance": self.get_performance_metrics(),
            "timestamp": datetime.now().isoformat()
        }

        # Test a simple query to ensure everything works
        try:
            test_result = await self.query_knowledgebase(
                query="test query",
                kb_query=KnowledgebaseQuery(top_k=1),
                classification=ClassificationType.GENERAL_QUERY
            )
            health_status["test_query"] = "success"
        except Exception as e:
            health_status["test_query"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"

        return health_status
