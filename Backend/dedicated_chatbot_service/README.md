# 🤖 Dedicated Chatbot Service

A standalone chatbot service running on port 8001, separated from other backend services to avoid conflicts and ensure reliable chat functionality.

## 🚀 Quick Start

### 1. Start the Service
```bash
cd Backend/dedicated_chatbot_service
start_chatbot.bat
```

### 2. Verify Service is Running
- Health Check: http://localhost:8001/health
- Test Endpoint: http://localhost:8001/test
- API Documentation: http://localhost:8001/docs

### 3. Test the Service
```bash
python test_chatbot.py
```

## 📡 API Endpoints

### POST /chatpost
Send a chat message to the service.

**Parameters:**
- `user_id` (query parameter): User identifier (default: "guest-user")

**Request Body:**
```json
{
  "message": "Hello, how are you?",
  "llm": "grok",
  "type": "chat_message"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Query received and stored",
  "data": {
    "id": "message_id",
    "user_id": "guest-user",
    "timestamp": "2025-07-14T..."
  }
}
```

### GET /chatbot
Get AI response for the latest user message.

**Parameters:**
- `user_id` (query parameter): User identifier (default: "guest-user")

**Response (Success):**
```json
{
  "_id": "message_id",
  "query": "Hello, how are you?",
  "response": {
    "message": "Hello! I'm doing well, thank you for asking...",
    "timestamp": "2025-07-14T...",
    "type": "chat_response",
    "user_id": "guest-user",
    "llm_model": "grok"
  }
}
```

**Response (No Queries):**
```json
{
  "error": "No queries yet"
}
```

### GET /chat-history
Get chat history for a user.

**Parameters:**
- `user_id` (query parameter): User identifier (default: "guest-user")
- `limit` (query parameter): Maximum number of messages (default: 50)

**Response:**
```json
{
  "user_id": "guest-user",
  "messages": [
    {
      "id": "message_id",
      "user_message": "Hello",
      "ai_response": "Hello! How can I help you?",
      "timestamp": "2025-07-14T...",
      "llm_model": "grok"
    }
  ],
  "total": 1
}
```

### GET /health
Service health check.

**Response:**
```json
{
  "status": "healthy",
  "service": "Dedicated Chatbot Service",
  "port": 8001,
  "mongodb": "connected",
  "llm_providers": {
    "groq": true,
    "openai": false,
    "fallback": true
  },
  "timestamp": "2025-07-14T..."
}
```

## 🔧 Configuration

### Environment Variables
The service uses environment variables from `../Base_backend/.env`:

- `GROQ_API_KEY`: Groq API key for LLM responses
- `OPENAI_API_KEY`: OpenAI API key (fallback)
- `MONGO_URI`: MongoDB connection string

### MongoDB Collection
The service uses a dedicated collection `dedicated_chat_messages` in the `gurukul` database to avoid conflicts with other services.

## 🏗️ Architecture

### Flow
1. Frontend sends POST request to `/chatpost` with user message
2. Service stores message in MongoDB with `status: "pending"`
3. Frontend sends GET request to `/chatbot` to get AI response
4. Service finds pending message, generates AI response using LLM service
5. Service updates message with response and `status: "completed"`
6. Service returns response to frontend

### LLM Integration
- Primary: Groq API (llama3-8b-8192)
- Fallback: OpenAI API (gpt-3.5-turbo)
- Final Fallback: Rule-based responses

### Error Handling
- "No queries yet": No pending messages found for user
- LLM failures: Automatic fallback to alternative providers
- MongoDB errors: Proper error responses with details

## 🧪 Testing

### Manual Testing
```bash
# Test health
curl http://localhost:8001/health

# Send message
curl -X POST "http://localhost:8001/chatpost?user_id=test-user" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "llm": "grok"}'

# Get response
curl "http://localhost:8001/chatbot?user_id=test-user"
```

### Automated Testing
```bash
python test_chatbot.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Port 8001 already in use**
   - The startup script automatically kills existing processes on port 8001
   - Manually check: `netstat -ano | findstr :8001`

2. **MongoDB connection failed**
   - Verify MONGO_URI in `.env` file
   - Check MongoDB service is running

3. **LLM API errors**
   - Verify GROQ_API_KEY in `.env` file
   - Check API key validity and rate limits

4. **"No queries yet" error persists**
   - Check if messages are being stored in MongoDB
   - Verify user_id consistency between chatpost and chatbot calls
   - Check service logs for processing errors

### Logs
The service provides detailed logging for debugging:
- MongoDB connection status
- Message storage and retrieval
- LLM API calls and responses
- Error details

## 🔄 Integration with Frontend

The frontend should:
1. Send POST to `/chatpost` with user message
2. Wait 1-2 seconds for processing
3. Send GET to `/chatbot` to retrieve AI response
4. Handle "No queries yet" error with retry logic
5. Display AI response to user

## 📦 Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- PyMongo: MongoDB driver
- Pydantic: Data validation
- python-dotenv: Environment variables
- requests: HTTP client for LLM APIs
