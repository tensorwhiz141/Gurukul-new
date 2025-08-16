# Memory Management System Architecture

## Overview

The Memory Management System provides a comprehensive API for storing, retrieving, and managing memory chunks and interactions for persona-based AI agents. It integrates with the existing MongoDB infrastructure and supports both persona-specific memory retrieval and recent interactions for chain-of-thought processing.

## Architecture Components

### 1. Data Models
- **MemoryChunk**: Core memory storage unit with content, metadata, and persona associations
- **Interaction**: User-agent interaction records with context and response data
- **PersonaMemory**: Aggregated memory context for specific personas
- **MemoryMetadata**: Additional context and tagging information

### 2. API Endpoints

#### Memory Storage
- `POST /memory` - Store memory chunks/interactions
- `PUT /memory/{memory_id}` - Update existing memory
- `DELETE /memory/{memory_id}` - Delete memory (soft delete with retention)

#### Memory Retrieval
- `GET /memory?persona={persona_id}` - Retrieve all memory for a persona
- `GET /memory?limit={number}` - Retrieve recent interactions
- `GET /memory?user_id={user_id}&limit={number}` - User-specific recent memories
- `GET /memory/{memory_id}` - Retrieve specific memory by ID

#### Memory Search and Filtering
- `GET /memory/search?query={text}` - Semantic search across memories
- `GET /memory/filter?tags={tag1,tag2}` - Filter by metadata tags
- `GET /memory/context?persona={persona_id}&topic={topic}` - Context-aware retrieval

### 3. Database Collections

#### memory_chunks
```json
{
  "_id": "ObjectId",
  "memory_id": "uuid",
  "user_id": "string",
  "persona_id": "string",
  "content": "string",
  "content_type": "text|interaction|context|reflection",
  "metadata": {
    "tags": ["string"],
    "importance": "number (1-10)",
    "topic": "string",
    "context_type": "string",
    "source": "string"
  },
  "timestamp": "ISO datetime",
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime",
  "is_active": "boolean",
  "embedding": "vector (optional for semantic search)"
}
```

#### memory_interactions
```json
{
  "_id": "ObjectId",
  "interaction_id": "uuid",
  "user_id": "string",
  "persona_id": "string",
  "user_message": "string",
  "agent_response": "string",
  "context": {
    "session_id": "string",
    "conversation_turn": "number",
    "domain": "string",
    "intent": "string"
  },
  "metadata": {
    "response_time": "number",
    "confidence": "number",
    "feedback": "string",
    "tags": ["string"]
  },
  "timestamp": "ISO datetime",
  "created_at": "ISO datetime",
  "is_active": "boolean"
}
```

#### persona_memory_index
```json
{
  "_id": "ObjectId",
  "persona_id": "string",
  "user_id": "string",
  "memory_summary": "string",
  "total_memories": "number",
  "last_interaction": "ISO datetime",
  "memory_categories": {
    "preferences": "number",
    "facts": "number",
    "interactions": "number",
    "context": "number"
  },
  "updated_at": "ISO datetime"
}
```

### 4. Security and Authentication

#### API Key Authentication
- Bearer token authentication for all endpoints
- Rate limiting: 1000 requests/hour per API key
- IP-based rate limiting for additional security

#### Data Privacy
- User data encryption at rest
- Secure memory deletion with configurable retention
- GDPR compliance for memory data export/deletion

### 5. Performance Considerations

#### Indexing Strategy
- Compound index on (user_id, persona_id, timestamp)
- Text index on content for search functionality
- TTL index for automatic cleanup of old memories

#### Caching
- Redis cache for frequently accessed persona memories
- Memory context caching for active sessions
- Query result caching with 5-minute TTL

#### Pagination
- Cursor-based pagination for large result sets
- Configurable page sizes (default: 50, max: 500)
- Efficient sorting by timestamp and relevance

## Integration Points

### Existing Systems
- **MongoDB**: Extends existing collections (user_data, chat_history)
- **FastAPI**: Integrates with current API structure
- **Authentication**: Uses existing auth patterns
- **Logging**: Extends current logging infrastructure

### Frontend Integration
- **React Components**: Memory context providers
- **API Hooks**: RTK Query integration for memory operations
- **Real-time Updates**: WebSocket support for live memory updates

## Configuration

### Environment Variables
```bash
MEMORY_DB_NAME=gurukul_memory
MEMORY_RETENTION_DAYS=365
MEMORY_MAX_SIZE_MB=100
MEMORY_CACHE_TTL=300
MEMORY_API_RATE_LIMIT=1000
```

### Feature Flags
- `ENABLE_SEMANTIC_SEARCH`: Vector-based memory search
- `ENABLE_MEMORY_COMPRESSION`: Automatic old memory summarization
- `ENABLE_REAL_TIME_SYNC`: WebSocket memory synchronization

## Monitoring and Analytics

### Metrics
- Memory storage/retrieval latency
- Memory usage per persona/user
- API endpoint usage statistics
- Error rates and types

### Health Checks
- Database connectivity
- Memory collection health
- Cache performance
- API response times

## Future Enhancements

### Phase 2 Features
- Semantic memory search with embeddings
- Automatic memory importance scoring
- Cross-persona memory sharing
- Memory compression and archival

### Phase 3 Features
- Federated memory across multiple domains
- AI-powered memory summarization
- Predictive memory pre-loading
- Advanced analytics and insights
