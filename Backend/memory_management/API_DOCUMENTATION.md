# Memory Management API Documentation

## Overview

The Memory Management API provides comprehensive endpoints for storing, retrieving, and managing memory chunks and interactions for persona-based AI agents. This API supports the Unified Agent Mind system with domain-specific memory capabilities.

## Base URL

- **Development**: `http://localhost:8003`
- **Production**: `https://api.gurukul.com` (configure as needed)

## Authentication

All endpoints require Bearer token authentication using API keys.

### Headers
```
Authorization: Bearer <your_api_key>
Content-Type: application/json
```

### API Keys
Contact your system administrator to obtain API keys. Different keys are available for:
- Development environment
- Production environment  
- Testing environment

### Rate Limiting
- **Hourly Limit**: 1000 requests per hour per API key
- **Burst Limit**: 100 requests per minute per API key
- **IP Limit**: 200 requests per 5 minutes per IP address

Rate limit headers are included in responses:
```
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1640995200
```

## API Endpoints

### Health Check

#### GET /memory/health
Check API health and database connectivity.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T12:00:00Z",
  "database": "connected"
}
```

### Memory Storage

#### POST /memory
Store a new memory chunk with content, metadata, and persona association.

**Request Body:**
```json
{
  "user_id": "user123",
  "persona_id": "financial_advisor",
  "content": "User prefers conservative investment strategies",
  "content_type": "preference",
  "metadata": {
    "tags": ["investment", "preference", "conservative"],
    "importance": 7,
    "topic": "investment_strategy",
    "context_type": "user_preference",
    "source": "user_input",
    "confidence": 0.95
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Memory chunk created successfully",
  "data": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8003/memory" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "persona_id": "financial_advisor",
    "content": "User prefers conservative investment strategies",
    "content_type": "preference"
  }'
```

#### POST /memory/interaction
Store a new user-agent interaction with context and metadata.

**Request Body:**
```json
{
  "user_id": "user123",
  "persona_id": "financial_advisor",
  "user_message": "What's the best investment strategy for me?",
  "agent_response": "Based on your profile, I recommend a conservative portfolio with 60% bonds and 40% stocks.",
  "context": {
    "session_id": "session_456",
    "conversation_turn": 3,
    "domain": "finance",
    "intent": "investment_advice"
  },
  "metadata": {
    "response_time": 1.2,
    "confidence": 0.89,
    "model_used": "gpt-4"
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Interaction created successfully",
  "data": {
    "interaction_id": "660e8400-e29b-41d4-a716-446655440001"
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

### Memory Retrieval

#### GET /memory?persona={persona_id}
Retrieve all memories for a specific persona with filtering and pagination.

**Query Parameters:**
- `persona` (required): Persona identifier
- `user_id` (optional): Filter by user ID
- `limit` (optional, default=50, max=500): Number of results per page
- `offset` (optional, default=0): Number of results to skip
- `content_type` (optional): Filter by content types (can specify multiple)
- `min_importance` (optional): Minimum importance level (1-10)

**Example Request:**
```
GET /memory?persona=financial_advisor&user_id=user123&limit=20&content_type=preference&content_type=fact
```

**Response (200 OK):**
```json
{
  "memories": [
    {
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user123",
      "persona_id": "financial_advisor",
      "content": "User prefers conservative investment strategies",
      "content_type": "preference",
      "metadata": {
        "tags": ["investment", "preference", "conservative"],
        "importance": 7,
        "topic": "investment_strategy",
        "context_type": "user_preference",
        "source": "user_input",
        "confidence": 0.95,
        "related_memories": []
      },
      "timestamp": "2023-12-01T12:00:00Z",
      "created_at": "2023-12-01T12:00:00Z",
      "updated_at": "2023-12-01T12:00:00Z",
      "is_active": true
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 20,
  "has_next": false,
  "has_previous": false
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8003/memory?persona=financial_advisor&limit=10" \
  -H "Authorization: Bearer your_api_key"
```

#### GET /memory?limit={number}&recent_interactions=true
Retrieve recent interactions for chain-of-thought processing.

**Query Parameters:**
- `limit` (required): Maximum number of interactions (1-500)
- `recent_interactions` (required): Must be `true`
- `user_id` (optional): Filter by user ID
- `persona` (optional): Filter by persona ID

**Example Request:**
```
GET /memory?limit=10&recent_interactions=true&user_id=user123
```

**Response (200 OK):**
```json
{
  "interactions": [
    {
      "interaction_id": "660e8400-e29b-41d4-a716-446655440001",
      "user_id": "user123",
      "persona_id": "financial_advisor",
      "user_message": "What's the best investment strategy for me?",
      "agent_response": "Based on your profile, I recommend a conservative portfolio.",
      "context": {
        "session_id": "session_456",
        "conversation_turn": 3,
        "domain": "finance",
        "intent": "investment_advice",
        "previous_context": null
      },
      "metadata": {
        "response_time": 1.2,
        "confidence": 0.89,
        "feedback": null,
        "tags": [],
        "model_used": "gpt-4"
      },
      "timestamp": "2023-12-01T12:00:00Z",
      "created_at": "2023-12-01T12:00:00Z",
      "is_active": true
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 10,
  "has_next": false,
  "has_previous": false
}
```

#### GET /memory/{memory_id}
Retrieve a specific memory chunk by its ID.

**Path Parameters:**
- `memory_id`: Unique memory identifier

**Response (200 OK):**
```json
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "persona_id": "financial_advisor",
  "content": "User prefers conservative investment strategies",
  "content_type": "preference",
  "metadata": {
    "tags": ["investment", "preference", "conservative"],
    "importance": 7,
    "topic": "investment_strategy",
    "context_type": "user_preference",
    "source": "user_input",
    "confidence": 0.95,
    "related_memories": []
  },
  "timestamp": "2023-12-01T12:00:00Z",
  "created_at": "2023-12-01T12:00:00Z",
  "updated_at": "2023-12-01T12:00:00Z",
  "is_active": true
}
```

#### GET /memory/persona/{persona_id}/summary
Get a summary of all memories for a specific persona.

**Path Parameters:**
- `persona_id`: Persona identifier

**Query Parameters:**
- `user_id` (optional): Filter by user ID

**Response (200 OK):**
```json
{
  "persona_id": "financial_advisor",
  "user_id": "user123",
  "total_memories": 25,
  "memory_categories": {
    "preference": 8,
    "fact": 12,
    "interaction": 3,
    "context": 2
  },
  "last_interaction": "2023-12-01T12:00:00Z",
  "importance_distribution": {
    "1": 2,
    "2": 3,
    "3": 5,
    "4": 8,
    "5": 4,
    "6": 2,
    "7": 1
  },
  "recent_topics": ["investment_strategy", "risk_assessment", "portfolio_management"],
  "updated_at": "2023-12-01T12:00:00Z"
}
```

### Memory Search

#### GET /memory/search
Search memories using text query with optional filters.

**Query Parameters:**
- `query` (required): Search query text
- `persona_id` (optional): Filter by persona ID
- `user_id` (optional): Filter by user ID
- `content_type` (optional): Filter by content types
- `limit` (optional, default=20, max=100): Maximum results

**Example Request:**
```
GET /memory/search?query=investment strategy&persona_id=financial_advisor&limit=5
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user123",
      "persona_id": "financial_advisor",
      "content": "User prefers conservative investment strategies",
      "content_type": "preference",
      "metadata": {
        "tags": ["investment", "preference", "conservative"],
        "importance": 7,
        "topic": "investment_strategy"
      },
      "timestamp": "2023-12-01T12:00:00Z",
      "created_at": "2023-12-01T12:00:00Z",
      "updated_at": "2023-12-01T12:00:00Z",
      "is_active": true
    }
  ],
  "query": "investment strategy",
  "total_results": 1,
  "search_time": 0.045,
  "suggestions": []
}
```

### Memory Management

#### PUT /memory/{memory_id}
Update an existing memory chunk.

**Path Parameters:**
- `memory_id`: Memory identifier

**Request Body:**
```json
{
  "content": "Updated content",
  "content_type": "fact",
  "metadata": {
    "tags": ["updated", "fact"],
    "importance": 8,
    "topic": "updated_topic"
  },
  "is_active": true
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Memory updated successfully",
  "data": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

#### DELETE /memory/{memory_id}
Delete a memory chunk (soft delete by default).

**Path Parameters:**
- `memory_id`: Memory identifier

**Query Parameters:**
- `hard_delete` (optional, default=false): Permanently delete if true

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Memory deactivated successfully",
  "data": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000",
    "hard_delete": false
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

## Data Models

### Content Types
- `text`: General text content
- `interaction`: User-agent interaction
- `context`: Contextual information
- `reflection`: Agent reflection or analysis
- `preference`: User preference
- `fact`: Factual information

### Importance Levels
Scale from 1 (very low) to 10 (essential):
- 1-2: Very low importance
- 3-4: Low importance
- 5-6: Moderate importance
- 7-8: High importance
- 9-10: Critical/Essential

### Memory Metadata Schema
```json
{
  "tags": ["string"],
  "importance": 1-10,
  "topic": "string",
  "context_type": "string",
  "source": "string",
  "confidence": 0.0-1.0,
  "related_memories": ["memory_id"]
}
```

### Interaction Context Schema
```json
{
  "session_id": "string",
  "conversation_turn": "number",
  "domain": "string",
  "intent": "string",
  "previous_context": "string"
}
```

## Error Handling

### HTTP Status Codes
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format
```json
{
  "error": "ERROR_TYPE",
  "message": "Human-readable error message",
  "details": [
    {
      "field": "field_name",
      "message": "Field-specific error message",
      "code": "ERROR_CODE"
    }
  ],
  "timestamp": "2023-12-01T12:00:00Z",
  "request_id": "req_123456"
}
```

### Common Error Examples

#### Invalid API Key (401)
```json
{
  "error": "UNAUTHORIZED",
  "message": "Invalid API key",
  "details": [],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

#### Rate Limit Exceeded (429)
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded",
  "details": [],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

#### Validation Error (400)
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [
    {
      "field": "content",
      "message": "Content cannot be empty",
      "code": "REQUIRED_FIELD"
    }
  ],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

## Code Examples

### Python Example
```python
import requests

# Configuration
API_BASE_URL = "http://localhost:8003"
API_KEY = "your_api_key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Store a memory
memory_data = {
    "user_id": "user123",
    "persona_id": "financial_advisor",
    "content": "User prefers conservative investments",
    "content_type": "preference",
    "metadata": {
        "tags": ["investment", "preference"],
        "importance": 7,
        "topic": "investment_strategy"
    }
}

response = requests.post(
    f"{API_BASE_URL}/memory",
    json=memory_data,
    headers=headers
)

if response.status_code == 201:
    result = response.json()
    memory_id = result["data"]["memory_id"]
    print(f"Memory created: {memory_id}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# Retrieve memories for a persona
response = requests.get(
    f"{API_BASE_URL}/memory?persona=financial_advisor&limit=10",
    headers=headers
)

if response.status_code == 200:
    memories = response.json()
    print(f"Found {memories['total_count']} memories")
    for memory in memories['memories']:
        print(f"- {memory['content'][:50]}...")
```

### JavaScript Example
```javascript
const API_BASE_URL = 'http://localhost:8003';
const API_KEY = 'your_api_key';

const headers = {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
};

// Store a memory
async function storeMemory() {
    const memoryData = {
        user_id: 'user123',
        persona_id: 'financial_advisor',
        content: 'User prefers conservative investments',
        content_type: 'preference',
        metadata: {
            tags: ['investment', 'preference'],
            importance: 7,
            topic: 'investment_strategy'
        }
    };

    try {
        const response = await fetch(`${API_BASE_URL}/memory`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(memoryData)
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Memory created:', result.data.memory_id);
        } else {
            console.error('Error:', response.status, await response.text());
        }
    } catch (error) {
        console.error('Network error:', error);
    }
}

// Retrieve recent interactions
async function getRecentInteractions() {
    try {
        const response = await fetch(
            `${API_BASE_URL}/memory?limit=5&recent_interactions=true`,
            { headers: headers }
        );

        if (response.ok) {
            const interactions = await response.json();
            console.log(`Found ${interactions.total_count} interactions`);
            interactions.interactions.forEach(interaction => {
                console.log(`User: ${interaction.user_message}`);
                console.log(`Agent: ${interaction.agent_response}`);
            });
        }
    } catch (error) {
        console.error('Error:', error);
    }
}
```

## Environment Configuration

### Environment Variables
```bash
# Database Configuration
MONGODB_URL=mongodb://localhost:27017/
MEMORY_DB_NAME=gurukul
MEMORY_RETENTION_DAYS=365

# API Configuration
MEMORY_API_KEYS=key1:user1,key2:user2
MEMORY_RATE_LIMIT_REQUESTS=1000
MEMORY_RATE_LIMIT_WINDOW=3600
MEMORY_RATE_LIMIT_BURST=100

# Security Configuration
MEMORY_MAX_SIZE_MB=100
MEMORY_CACHE_TTL=300
```

### Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8003

CMD ["uvicorn", "memory_management.api:app", "--host", "0.0.0.0", "--port", "8003"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  memory-api:
    build: .
    ports:
      - "8003:8003"
    environment:
      - MONGODB_URL=mongodb://mongo:27017/
      - MEMORY_DB_NAME=gurukul
    depends_on:
      - mongo

  mongo:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

## Testing

### Health Check
```bash
curl -X GET "http://localhost:8003/memory/health"
```

### API Documentation
- **Swagger UI**: `http://localhost:8003/memory/docs`
- **ReDoc**: `http://localhost:8003/memory/redoc`

### Performance Considerations
- Use pagination for large result sets
- Implement caching for frequently accessed data
- Monitor database indexes for optimal query performance
- Consider rate limiting based on your usage patterns

## Support

For technical support or questions about the Memory Management API:
- Check the interactive documentation at `/memory/docs`
- Review error messages and status codes
- Ensure proper authentication and rate limiting compliance
- Contact your system administrator for API key issues
```
