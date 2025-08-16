# Agent Mind-Auth-Memory Link

A comprehensive AI agent system that binds voice (TTS), identity (Supabase Auth), and memory (MongoDB) into a unified API.

## 🎯 Features

- **🔐 Supabase Authentication**: JWT token verification and user management
- **🧠 MongoDB Memory**: Persistent chat history and session storage
- **🤖 Agent Integration**: Seamless integration with Vedant's /ask-agent API
- **🔊 TTS Support**: Optional text-to-speech audio generation
- **📊 RESTful API**: Clean, documented API endpoints
- **🧪 Comprehensive Tests**: Full test suite with pytest

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Auth Layer    │    │   Memory Layer  │
│   (Rishabh)     │◄──►│   (Supabase)    │◄──►│   (MongoDB)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Agent Mind-Auth-      │
                    │   Memory Link (This)    │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Vedant's Agent API    │
                    │   (/ask-agent)          │
                    └─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (running locally or remote)
- Supabase project with JWT secret

### Installation

1. **Clone and setup**:
   ```bash
   cd "Desktop\augmed kamal"
   python setup.py
   ```

2. **Configure environment**:
   Update `.env` file with your actual values:
   ```env
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_anon_key_here
   SUPABASE_JWT_SECRET=your_supabase_jwt_secret_here
   
   MONGODB_URI=mongodb://localhost:27017/
   MONGODB_DATABASE=agent_memory
   
   AGENT_API_URL=http://localhost:8000/ask-agent
   AGENT_API_KEY=your_agent_api_key_here
   ```

3. **Start the application**:
   ```bash
   python main.py
   ```

4. **Test the API**:
   Visit http://localhost:8000/docs for interactive API documentation

## 📡 API Endpoints

### Authentication Required

All endpoints require a valid Supabase JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

### Core Endpoints

#### `POST /api/v1/chat`
Send a message to the AI agent with optional chat history and TTS.

**Request**:
```json
{
  "message": "Hello, how are you?",
  "include_history": true,
  "max_history_messages": 10,
  "tts_enabled": false
}
```

**Response**:
```json
{
  "response": "I'm doing well, thank you for asking!",
  "user_id": "user-123",
  "timestamp": "2025-06-20T13:14:15.123456",
  "audio_url": null,
  "metadata": {
    "model": "gpt-4",
    "tokens": 150
  }
}
```

#### `POST /api/v1/save_progress`
Save current chat session to memory.

**Request**:
```json
{
  "chat_history": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-06-20T13:14:15.123456"
    },
    {
      "role": "assistant", 
      "content": "Hi there!",
      "timestamp": "2025-06-20T13:14:16.123456"
    }
  ],
  "session_metadata": {
    "session_type": "conversation"
  }
}
```

#### `GET /api/v1/chat_history`
Retrieve user's chat history with pagination.

**Query Parameters**:
- `limit`: Number of messages to return (default: 50)
- `skip`: Number of messages to skip (default: 0)

#### `GET /api/v1/user_session`
Get user's current session data.

#### `DELETE /api/v1/user_data`
Delete all user data (GDPR compliance).

### Health Check

#### `GET /health`
Check system health status.

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_auth.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | Yes |
| `SUPABASE_KEY` | Supabase anon key | Yes |
| `SUPABASE_JWT_SECRET` | JWT secret for token verification | Yes |
| `MONGODB_URI` | MongoDB connection string | Yes |
| `MONGODB_DATABASE` | Database name | No (default: agent_memory) |
| `AGENT_API_URL` | Vedant's agent API endpoint | Yes |
| `AGENT_API_KEY` | API key for agent service | Yes |
| `TTS_ENABLED` | Enable TTS functionality | No (default: false) |
| `TTS_API_URL` | TTS service endpoint | No |

### MongoDB Collections

The system creates two main collections:

1. **chat_history**: Individual chat messages
   ```json
   {
     "user_id": "string",
     "message": "string", 
     "response": "string",
     "timestamp": "datetime",
     "metadata": "object"
   }
   ```

2. **user_sessions**: User session data
   ```json
   {
     "user_id": "string",
     "chat_history": "array",
     "last_seen": "datetime", 
     "session_metadata": "object"
   }
   ```

## 🔗 Integration Points

### Frontend Integration (Rishabh)

The frontend should:
1. Obtain JWT tokens from Supabase Auth
2. Include tokens in all API requests
3. Handle audio URLs for TTS playback
4. Manage chat history display

### Agent API Integration (Vedant)

Expected `/ask-agent` API format:
```json
{
  "query": "user message",
  "user_id": "user-123",
  "context": "formatted chat history"
}
```

## 🛠️ Development

### Project Structure

```
├── auth/                 # Supabase authentication
├── memory/              # MongoDB integration  
├── api/                 # API endpoints and handlers
├── models/              # Pydantic schemas
├── tests/               # Test suite
├── main.py              # FastAPI application
├── requirements.txt     # Dependencies
└── .env.example         # Environment template
```

### Adding New Features

1. Create feature branch
2. Add implementation in appropriate module
3. Add comprehensive tests
4. Update documentation
5. Test integration points

## 📝 License

This project is part of the Agent Mind-Auth-Memory Link system.

## 🤝 Team

- **Aakash**: Backend integration (this system)
- **Rishabh**: Frontend integration
- **Vedant**: Agent API development

---

**Status**: ✅ Phase 1 Complete - Ready for integration testing
