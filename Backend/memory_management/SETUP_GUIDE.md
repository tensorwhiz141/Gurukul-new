# Memory Management System - Setup Guide

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- MongoDB 5.0 or higher
- Git

### 2. Installation

```bash
# Clone the repository (if not already done)
cd Backend/memory_management

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env file with your settings
# At minimum, update MONGODB_URL and MEMORY_API_KEYS
```

### 3. Environment Configuration

Edit the `.env` file with your specific settings:

```bash
# Database
MONGODB_URL=mongodb://localhost:27017/
MEMORY_DB_NAME=gurukul

# API Keys (generate secure keys for production)
MEMORY_API_KEYS=your_dev_key:dev_user,your_prod_key:prod_user

# Server settings
MEMORY_API_HOST=0.0.0.0
MEMORY_API_PORT=8003
```

### 4. Start the Server

```bash
# Option 1: Using the startup script
python run_server.py

# Option 2: Direct uvicorn command
uvicorn api:app --host 0.0.0.0 --port 8003 --reload

# Option 3: Using the module
python -m memory_management.api
```

### 5. Verify Installation

```bash
# Check health endpoint
curl http://localhost:8003/memory/health

# View API documentation
# Open http://localhost:8003/memory/docs in your browser
```

## Integration with Existing System

### 1. Add to Existing FastAPI Application

```python
from fastapi import FastAPI
from memory_management.api import app as memory_app

# Your existing app
app = FastAPI()

# Mount memory management API
app.mount("/memory", memory_app)

# Or include routers
from memory_management.api import router
app.include_router(router, prefix="/memory", tags=["memory"])
```

### 2. Database Integration

The system integrates with your existing MongoDB setup:

```python
# Use existing MongoDB connection
from memory_management.database import MemoryDatabase

# Initialize with your existing client
db = MemoryDatabase()
# The system will use the configured MONGODB_URL
```

### 3. Authentication Integration

```python
# Integrate with existing auth system
from memory_management.auth import verify_api_key
from fastapi import Depends

@app.get("/protected-endpoint")
async def protected_route(current_user: str = Depends(verify_api_key)):
    # Your existing logic
    pass
```

## Usage Examples

### 1. Basic Memory Storage

```python
import httpx

# Store a memory
memory_data = {
    "user_id": "user123",
    "persona_id": "financial_advisor",
    "content": "User prefers conservative investments",
    "content_type": "preference",
    "metadata": {
        "tags": ["investment", "conservative"],
        "importance": 8,
        "topic": "investment_strategy"
    }
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8003/memory",
        json=memory_data,
        headers={"Authorization": "Bearer your_api_key"}
    )
    result = response.json()
    memory_id = result["data"]["memory_id"]
```

### 2. Retrieve Persona Memories

```python
# Get all memories for a persona
async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://localhost:8003/memory?persona=financial_advisor&limit=10",
        headers={"Authorization": "Bearer your_api_key"}
    )
    memories = response.json()
```

### 3. Chain-of-Thought Context

```python
# Get recent interactions for context
async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://localhost:8003/memory?limit=5&recent_interactions=true&user_id=user123",
        headers={"Authorization": "Bearer your_api_key"}
    )
    interactions = response.json()
    
    # Use interactions for context in your AI agent
    context = "\n".join([
        f"User: {i['user_message']}\nAgent: {i['agent_response']}"
        for i in interactions['interactions']
    ])
```

## Testing

### 1. Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
python -m pytest test_api.py -v

# Run with coverage
python -m pytest test_api.py --cov=memory_management
```

### 2. Manual Testing

```bash
# Run the example scenarios
python examples.py

# Use cURL examples
bash curl_examples.sh  # On Unix systems
# Or run individual curl commands from the file
```

### 3. Load Testing

```bash
# Install load testing tools
pip install locust

# Create a simple load test
# See examples.py for API usage patterns
```

## Production Deployment

### 1. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8003

CMD ["python", "run_server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  memory-api:
    build: .
    ports:
      - "8003:8003"
    environment:
      - MONGODB_URL=mongodb://mongo:27017/
      - MEMORY_DB_NAME=gurukul_prod
      - MEMORY_API_WORKERS=4
    depends_on:
      - mongo
  
  mongo:
    image: mongo:5.0
    volumes:
      - mongo_data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password

volumes:
  mongo_data:
```

### 2. Environment Variables for Production

```bash
# Production .env
MONGODB_URL=mongodb://username:password@mongo-cluster:27017/
MEMORY_DB_NAME=gurukul_production
MEMORY_API_WORKERS=4
MEMORY_API_RELOAD=false

# Secure API keys
MEMORY_API_KEYS=prod_key_1:user1,prod_key_2:user2

# Rate limiting
MEMORY_RATE_LIMIT_REQUESTS=5000
MEMORY_RATE_LIMIT_BURST=500

# Security
ENVIRONMENT=production
```

### 3. Monitoring and Logging

```python
# Add monitoring
import logging
from memory_management.api import app

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add health check endpoint monitoring
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    return response
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check MongoDB is running
   mongosh --eval "db.adminCommand('ping')"
   
   # Check connection string
   echo $MONGODB_URL
   ```

2. **Authentication Errors**
   ```bash
   # Verify API key format
   echo $MEMORY_API_KEYS
   
   # Test with curl
   curl -H "Authorization: Bearer your_key" http://localhost:8003/memory/health
   ```

3. **Import Errors**
   ```bash
   # Check Python path
   python -c "import memory_management; print('OK')"
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Port Already in Use**
   ```bash
   # Check what's using port 8003
   netstat -tulpn | grep 8003
   
   # Use different port
   export MEMORY_API_PORT=8004
   ```

### Performance Tuning

1. **Database Indexes**
   - The system automatically creates necessary indexes
   - Monitor slow queries in MongoDB logs
   - Consider additional indexes for your specific query patterns

2. **Rate Limiting**
   - Adjust rate limits based on your usage patterns
   - Monitor rate limit headers in responses
   - Implement caching for frequently accessed data

3. **Memory Usage**
   - Monitor memory usage with large datasets
   - Implement pagination for large result sets
   - Consider memory cleanup policies

## Support and Maintenance

### Regular Maintenance

1. **Database Cleanup**
   ```python
   # The system includes TTL indexes for automatic cleanup
   # Monitor database size and adjust retention policies
   ```

2. **Log Rotation**
   ```bash
   # Set up log rotation for memory_api.log
   # Monitor disk space usage
   ```

3. **Security Updates**
   ```bash
   # Regularly update dependencies
   pip install -r requirements.txt --upgrade
   
   # Rotate API keys periodically
   # Monitor for security vulnerabilities
   ```

### Monitoring Checklist

- [ ] API response times
- [ ] Database connection health
- [ ] Memory usage and storage growth
- [ ] Rate limiting effectiveness
- [ ] Error rates and types
- [ ] Authentication failures

For additional support, refer to:
- API Documentation: `/memory/docs`
- Example code: `examples.py`
- Test cases: `test_api.py`
- cURL examples: `curl_examples.sh`
