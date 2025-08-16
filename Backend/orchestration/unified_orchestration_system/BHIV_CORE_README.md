# BHIV Core v0.9 - Complete Implementation

## Overview

BHIV Core is a comprehensive AI orchestration system that implements the complete flow:
**USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO**

This implementation ensures coherent communication between all system components and provides a robust, scalable foundation for the Gurukul educational platform.

## Architecture

### Core Components

1. **BHIV Core Orchestrator** (`bhiv_core_orchestrator.py`)
   - Main orchestration engine
   - Manages the complete processing pipeline
   - Coordinates between all components

2. **BHIV Alpha Router** (`bhiv_alpha_router.py`)
   - Intelligent agent routing and planning
   - Advanced query classification
   - Context-aware decision making

3. **BHIV Knowledgebase** (`bhiv_knowledgebase.py`)
   - Enhanced vector retrieval system
   - Qdrant and FAISS integration
   - Performance optimization and caching

4. **Voice & Video Pipeline** (`bhiv_voice_video_pipeline.py`)
   - Multi-provider TTS integration
   - Voice response generation
   - Future video capabilities

5. **Session Manager** (`bhiv_session_manager.py`)
   - Comprehensive session lifecycle management
   - Interaction logging and analytics
   - Reward scoring system

6. **API Gateway** (`bhiv_api_gateway.py`)
   - Authentication and authorization
   - Rate limiting and security
   - External API management

## Schema Definition

### Core Schema (`bhiv_core_schema.py`)

The BHIV Core schema defines the complete data flow:

```python
{
  "session_id": "abc-1234-xyz",
  "timestamp": "2025-07-30T14:10:25Z",
  "user_info": {
    "user_id": "user_001",
    "auth_token": "jwt_or_api_key",
    "lang": "en-IN",
    "persona": "student",
    "permissions": ["read", "write"]
  },
  "input": {
    "raw_text": "How do I start learning AI from scratch?",
    "voice_enabled": true,
    "mode": "chat",
    "context": {
      "course": "AI/Seed/Level1",
      "previous_message_id": "msg_0003"
    }
  },
  "agent_directives": {
    "classification": "learning-query",
    "urgency": "normal",
    "trigger_module": "knowledgebase",
    "agent_name": "GuruAgent",
    "expected_output_format": "rich_text_with_tts"
  },
  "knowledgebase_query": {
    "use_vector": true,
    "vector_type": "qdrant",
    "top_k": 5,
    "include_sources": true,
    "query_embedding": "<vector>",
    "filters": {
      "lang": "en",
      "curriculum_tag": ["AI", "beginner"]
    }
  },
  "llm_request": {
    "model": "llama3",
    "prompt_style": "instruction",
    "max_tokens": 1024,
    "temperature": 0.3,
    "response_language": "en"
  },
  "output": {
    "response_text": "To start learning AI, begin with...",
    "confidence_score": 0.88,
    "sources": ["AI curriculum module 1", "Wikipedia (fallback)"],
    "voice_response_url": "https://vaani.uniguru.ai/audio/abc123.mp3",
    "followup_suggestions": [
      "Would you like to begin Module 1 now?",
      "Do you want to see related videos?"
    ]
  },
  "logs": {
    "agent_chain_log": ["classified: learning", "vector-search: 5 docs", "llm-response: success"],
    "reward_score": 1.0,
    "error_flag": false
  }
}
```

## Installation and Setup

### Prerequisites

```bash
# Python dependencies
pip install fastapi uvicorn pydantic
pip install langchain langchain-huggingface
pip install qdrant-client redis
pip install aiohttp aiofiles
pip install jwt pyttsx3

# Optional: Advanced TTS providers
pip install elevenlabs openai
```

### Environment Variables

```bash
# Core Configuration
JWT_SECRET_KEY=your-secret-key
REDIS_URL=redis://localhost:6379
QDRANT_URL=localhost:6333

# External APIs
OPENAI_API_KEY=your-openai-key
ELEVENLABS_API_KEY=your-elevenlabs-key
GEMINI_API_KEY=your-gemini-key

# Audio Configuration
AUDIO_BASE_URL=http://localhost:8010/audio
TTS_SERVICE_URL=http://localhost:8005
```

### Database Setup

```bash
# SQLite (default for development)
# Database will be created automatically

# For production, configure PostgreSQL or MySQL
DATABASE_URL=postgresql://user:password@localhost/bhiv_core
```

## Usage

### Starting the BHIV Core API

```python
# Start the BHIV Core API server
python bhiv_core_api.py
```

The API will be available at `http://localhost:8010`

### Basic Usage Example

```python
import asyncio
from bhiv_core_schema import create_bhiv_request, UserPersona, InputMode
from bhiv_core_orchestrator import BHIVCoreOrchestrator

async def example_usage():
    # Initialize orchestrator
    orchestrator = BHIVCoreOrchestrator()
    await orchestrator.initialize()
    
    # Create request
    request = create_bhiv_request(
        user_id="demo_user",
        raw_text="How do I learn machine learning?",
        voice_enabled=True,
        mode=InputMode.CHAT,
        persona=UserPersona.STUDENT,
        lang="en-IN"
    )
    
    # Process request
    response = await orchestrator.process_bhiv_request(request)
    
    # Use response
    print(f"Agent: {response.agent_directives.agent_name}")
    print(f"Response: {response.output.response_text}")
    print(f"Confidence: {response.output.confidence_score}")

# Run example
asyncio.run(example_usage())
```

### Frontend Integration

```javascript
// Using the BHIV Core API slice
import { useProcessBHIVRequestMutation } from '../api/bhivCoreApiSlice';
import { createQuickBHIVRequest } from '../api/bhivCoreApiSlice';

function ChatComponent() {
  const [processBHIVRequest] = useProcessBHIVRequestMutation();
  
  const handleQuery = async (query) => {
    const request = createQuickBHIVRequest("user123", query, {
      voiceEnabled: true,
      mode: "chat"
    });
    
    const response = await processBHIVRequest(request).unwrap();
    
    // Handle response
    console.log(response.output.response_text);
    if (response.output.voice_response_url) {
      // Play audio response
      const audio = new Audio(response.output.voice_response_url);
      audio.play();
    }
  };
}
```

## API Endpoints

### Core Endpoints

- `POST /api/v1/bhiv-core/process` - Main BHIV processing
- `POST /api/v1/bhiv-core/quick-query` - Quick query processing
- `POST /api/v1/bhiv-core/classify` - Query classification only
- `GET /api/v1/bhiv-core/session/{session_id}` - Session information
- `GET /api/v1/bhiv-core/stats` - System statistics
- `GET /health` - Health check

### Authentication

All endpoints support both JWT tokens and API keys:

```bash
# Using JWT token
curl -H "Authorization: Bearer <jwt_token>" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123", "query": "Hello"}' \
     http://localhost:8010/api/v1/bhiv-core/quick-query

# Using API key
curl -H "Authorization: Bearer bhiv-core-key" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123", "query": "Hello"}' \
     http://localhost:8010/api/v1/bhiv-core/quick-query
```

## Testing

### Unit Tests

```bash
# Run unit tests
python -m pytest test_bhiv_core.py -v
```

### Integration Tests

```bash
# Run comprehensive integration tests
python integration_test_bhiv.py
```

### Performance Testing

```bash
# Run performance benchmarks
python -c "
import asyncio
from integration_test_bhiv import BHIVIntegrationTester
async def perf_test():
    tester = BHIVIntegrationTester()
    await tester.test_component_initialization()
    await tester.test_performance_benchmarks()
asyncio.run(perf_test())
"
```

## Configuration

### Agent Configuration

Agents can be configured in the orchestrator:

```python
# Custom agent configuration
AGENT_CONFIG = {
    "EduMentor": {
        "classification": ClassificationType.LEARNING_QUERY,
        "trigger_module": TriggerModule.KNOWLEDGEBASE,
        "max_tokens": 1024,
        "temperature": 0.3
    },
    "WellnessBot": {
        "classification": ClassificationType.WELLNESS_QUERY,
        "trigger_module": TriggerModule.WELLNESS_BOT,
        "max_tokens": 512,
        "temperature": 0.5
    }
}
```

### Vector Store Configuration

```python
# Qdrant configuration
QDRANT_CONFIG = {
    "url": "localhost:6333",
    "collections": {
        "educational": {"size": 384, "distance": "cosine"},
        "wellness": {"size": 384, "distance": "cosine"},
        "vedas": {"size": 384, "distance": "cosine"}
    }
}
```

## Monitoring and Analytics

### Performance Metrics

The system tracks comprehensive metrics:

- Request processing times
- Classification accuracy
- Agent performance
- Cache hit rates
- Error rates
- User satisfaction scores

### Health Monitoring

```python
# Check system health
health_status = await orchestrator.health_check()
print(f"Status: {health_status['status']}")
print(f"Components: {health_status['components']}")
```

### Session Analytics

```python
# Get user analytics
analytics = await session_manager.get_user_analytics("user123")
print(f"Total sessions: {analytics['total_sessions']}")
print(f"Preferred agents: {analytics['preferred_agents']}")
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8010
CMD ["python", "bhiv_core_api.py"]
```

### Production Considerations

1. **Database**: Use PostgreSQL or MySQL for production
2. **Redis**: Configure Redis cluster for high availability
3. **Load Balancing**: Use nginx or similar for load balancing
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Logging**: Configure structured logging with ELK stack
6. **Security**: Implement proper JWT secret rotation

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   ```bash
   # Check Qdrant is running
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Redis Connection Failed**
   ```bash
   # Check Redis is running
   redis-server
   ```

3. **TTS Provider Issues**
   ```bash
   # Check API keys are set
   echo $OPENAI_API_KEY
   echo $ELEVENLABS_API_KEY
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure all integration tests pass

## License

This implementation is part of the Gurukul project and follows the project's licensing terms.

---

For more information, see the individual component documentation in each Python file.
