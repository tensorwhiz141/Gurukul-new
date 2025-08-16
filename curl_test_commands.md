# 🚀 Gurukul Learning Platform - cURL Test Commands

This document contains comprehensive cURL commands to test all the APIs in the Gurukul Learning Platform.

## 📋 Prerequisites

1. **Start all services** using the batch file:
   ```bash
   cd Backend
   start_all_services.bat
   ```

2. **Verify services are running** on their respective ports:
   - Base Backend: http://localhost:8000
   - API Data Service: http://localhost:8001
   - Financial Simulator: http://localhost:8002
   - Memory Management: http://localhost:8003
   - Augmed Kamal: http://localhost:8004
   - Subject Generation: http://localhost:8005
   - Karthikeya: http://localhost:5000
   - TTS Service: http://localhost:8006

## 🎯 Immediate Priorities - Minimum Viable Deployment

### Core End-to-End Flow
```
Login → Dashboard → Subject Select → Lesson View (Text + Media) → Quiz → Progress Update
```

### Database Schema (Core Tables)
- **Users** - User authentication and profiles
- **Subjects** - Available subjects/courses
- **SyllabusNodes** - Subject structure and organization
- **Lessons** - Individual lesson content
- **Quizzes** - Assessment questions and answers
- **Progress** - User learning progress tracking

## 🔧 Configuration

Set these environment variables for authentication:
```bash
export API_KEY="memory_api_key_dev"
export SUPABASE_JWT="your_supabase_jwt_token"
```

---

## 🎯 **MVD ENDPOINTS - MINIMUM VIABLE DEPLOYMENT**

### 1. Dashboard Endpoint
```bash
# Get user dashboard data
curl -X GET "http://localhost:8000/dashboard/user123" \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json"
```

### 2. Syllabus Endpoint
```bash
# Get syllabus structure for a subject
curl -X GET "http://localhost:8000/syllabus/math_001" \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json"
```

### 3. Lesson Endpoint
```bash
# Get specific lesson content
curl -X GET "http://localhost:8000/lesson/lesson_001" \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json"
```

### 4. Quiz Endpoint
```bash
# Get quiz for a lesson
curl -X GET "http://localhost:8000/quiz/lesson_001" \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json"
```

### 5. Lesson Complete Endpoint
```bash
# Mark lesson as complete
curl -X POST "http://localhost:8000/lesson-complete" \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "lesson_id": "lesson_001",
    "quiz_score": 85.5,
    "time_spent": 45,
    "completed_sections": ["section_1", "section_2"]
  }'
```

### 6. Complete MVD Flow Test

#### For Linux/Mac:
```bash
# Test the complete end-to-end flow
echo "=== Testing Complete MVD Flow ==="

# 1. Get dashboard
echo "1. Getting dashboard..."
curl -X GET "http://localhost:8000/dashboard/user123" \
  -H "Authorization: Bearer $SUPABASE_JWT" | jq '.'

# 2. Get subjects
echo "2. Getting subjects..."
curl -X GET "http://localhost:8000/subjects" | jq '.'

# 3. Get syllabus for first subject
echo "3. Getting syllabus..."
curl -X GET "http://localhost:8000/syllabus/math_001" \
  -H "Authorization: Bearer $SUPABASE_JWT" | jq '.'

# 4. Get lesson
echo "4. Getting lesson..."
curl -X GET "http://localhost:8000/lesson/lesson_001" \
  -H "Authorization: Bearer $SUPABASE_JWT" | jq '.'

# 5. Get quiz
echo "5. Getting quiz..."
curl -X GET "http://localhost:8000/quiz/lesson_001" \
  -H "Authorization: Bearer $SUPABASE_JWT" | jq '.'

# 6. Complete lesson
echo "6. Completing lesson..."
curl -X POST "http://localhost:8000/lesson-complete" \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "lesson_id": "lesson_001",
    "quiz_score": 85.5,
    "time_spent": 45,
    "completed_sections": ["section_1", "section_2"]
  }' | jq '.'

echo "=== MVD Flow Test Complete ==="
```

#### For Windows (PowerShell):
```powershell
# Test the complete end-to-end flow
Write-Host "=== Testing Complete MVD Flow ==="

# 1. Get dashboard
Write-Host "1. Getting dashboard..."
curl -X GET "http://localhost:8000/dashboard/user123" -H "Content-Type: application/json"

# 2. Get subjects
Write-Host "2. Getting subjects..."
curl -X GET "http://localhost:8000/subjects"

# 3. Get syllabus for first subject
Write-Host "3. Getting syllabus..."
curl -X GET "http://localhost:8000/syllabus/math_001" -H "Content-Type: application/json"

# 4. Get lesson
Write-Host "4. Getting lesson..."
curl -X GET "http://localhost:8000/lesson/lesson_001" -H "Content-Type: application/json"

# 5. Get quiz
Write-Host "5. Getting quiz..."
curl -X GET "http://localhost:8000/quiz/lesson_001" -H "Content-Type: application/json"

# 6. Complete lesson
Write-Host "6. Completing lesson..."
curl -X POST "http://localhost:8000/lesson-complete" -H "Content-Type: application/json" -d "{\"user_id\": \"user123\", \"lesson_id\": \"lesson_001\", \"quiz_score\": 85.5, \"time_spent\": 45}"

Write-Host "=== MVD Flow Test Complete ==="
```

#### For Windows (Command Prompt):
```cmd
REM Test the complete end-to-end flow
echo === Testing Complete MVD Flow ===

REM 1. Get dashboard
echo 1. Getting dashboard...
curl -X GET "http://localhost:8000/dashboard/user123" -H "Content-Type: application/json"

REM 2. Get subjects
echo 2. Getting subjects...
curl -X GET "http://localhost:8000/subjects"

REM 3. Get syllabus for first subject
echo 3. Getting syllabus...
curl -X GET "http://localhost:8000/syllabus/math_001" -H "Content-Type: application/json"

REM 4. Get lesson
echo 4. Getting lesson...
curl -X GET "http://localhost:8000/lesson/lesson_001" -H "Content-Type: application/json"

REM 5. Get quiz
echo 5. Getting quiz...
curl -X GET "http://localhost:8000/quiz/lesson_001" -H "Content-Type: application/json"

REM 6. Complete lesson
echo 6. Completing lesson...
curl -X POST "http://localhost:8000/lesson-complete" -H "Content-Type: application/json" -d "{\"user_id\": \"user123\", \"lesson_id\": \"lesson_001\", \"quiz_score\": 85.5, \"time_spent\": 45}"

echo === MVD Flow Test Complete ===
```

---

## 🏥 Health Checks

### Base Backend (Port 8000)
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Integration status
curl -X GET "http://localhost:8000/integration-status"
```

### API Data Service (Port 8001)
```bash
# Health check
curl -X GET "http://localhost:8001/health"

# Dummy subjects
curl -X GET "http://localhost:8001/subjects_dummy"

# Dummy lectures
curl -X GET "http://localhost:8001/lecture_dummy"

# Dummy tests
curl -X GET "http://localhost:8001/test_dummy"
```

### Financial Simulator (Port 8002)
```bash
# Health check
curl -X GET "http://localhost:8002/health"

# System metrics
curl -X GET "http://localhost:8002/metrics"

# Basic forecast (7 days)
curl -X GET "http://localhost:8002/forecast?days=7"

# Extended forecast (30 days)
curl -X GET "http://localhost:8002/forecast?days=30&format=json"

# Chart-ready forecast
curl -X GET "http://localhost:8002/forecast-json?days=5"
```

### Memory Management (Port 8003)
```bash
# Health check
curl -X GET "http://localhost:8003/memory/health"
```

### Augmed Kamal (Port 8004)
```bash
# Health check
curl -X GET "http://localhost:8004/health"
```

### Subject Generation (Port 8005)
```bash
# Health check
curl -X GET "http://localhost:8005/"

# LLM status
curl -X GET "http://localhost:8005/llm_status"
```

### Karthikeya (Port 5000)
```bash
# Health check
curl -X GET "http://localhost:5000/health"

# Available languages
curl -X GET "http://localhost:5000/languages"
```

### TTS Service (Port 8006)
```bash
# Health check
curl -X GET "http://localhost:8006/api/health"

# List audio files
curl -X GET "http://localhost:8006/api/list-audio-files"
```

---

## 📚 Educational Services

### Get Subjects and Lectures
```bash
# Get all subjects
curl -X GET "http://localhost:8000/subjects"

# Get lectures
curl -X GET "http://localhost:8000/lectures"

# Get tests
curl -X GET "http://localhost:8000/tests"
```

### Generate Lessons
```bash
# Generate lesson
curl -X GET "http://localhost:8000/generate_lesson"

# Generate enhanced lesson
curl -X POST "http://localhost:8000/lessons/enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Mathematics",
    "topic": "Algebra",
    "level": "intermediate",
    "language": "en"
  }'

# Generate lesson with TTS
curl -X POST "http://localhost:8005/lessons/generate-tts" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Mathematics",
    "topic": "Algebra",
    "language": "en"
  }'
```

### Subject Generation Service
```bash
# Generate lesson
curl -X GET "http://localhost:8005/generate_lesson"

# Create lesson
curl -X POST "http://localhost:8005/lessons" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Physics",
    "topic": "Mechanics",
    "level": "beginner",
    "language": "en"
  }'

# Get lesson by subject and topic
curl -X GET "http://localhost:8005/lessons/Mathematics/Algebra"

# Search lessons
curl -X GET "http://localhost:8005/search_lessons?query=algebra"

# Get lesson status
curl -X GET "http://localhost:8005/lessons/status/task_id_here"

# Get all tasks
curl -X GET "http://localhost:8005/lessons/tasks"
```

---

## 🤖 AI Chat Services

### Base Backend Chat
```bash
# Chat endpoint
curl -X POST "http://localhost:8000/chatpost" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is algebra?",
    "user_id": "user123",
    "context": "educational"
  }'

# Chatbot endpoint
curl -X GET "http://localhost:8000/chatbot?message=Hello&user_id=user123"
```

### API Data Service Chat
```bash
# Chat endpoint
curl -X POST "http://localhost:8001/chatpost" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain calculus",
    "user_id": "user123"
  }'

# Chatbot endpoint
curl -X GET "http://localhost:8001/chatbot?message=Hello&user_id=user123"
```

### Dedicated Chatbot Service
```bash
# Chat endpoint
curl -X POST "http://localhost:8007/chatpost" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how can you help me?",
    "user_id": "user123"
  }'

# Chatbot endpoint
curl -X GET "http://localhost:8007/chatbot?message=Hello&user_id=user123"

# Get chat history
curl -X GET "http://localhost:8007/chat-history?user_id=user123"

# Test endpoint
curl -X GET "http://localhost:8007/test"
```

---

## 💰 Financial Services

### Agent Scoring
```bash
# Score agent with light load
curl -X POST "http://localhost:8002/score-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "current_load": 10
  }'

# Score agent with heavy load
curl -X POST "http://localhost:8002/score-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_002",
    "current_load": 25
  }'

# Score agent with overload
curl -X POST "http://localhost:8002/score-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_003",
    "current_load": 35
  }'
```

### Workflow Simulation
```bash
# Simulate workflow
curl -X POST "http://localhost:8002/simulate-workflow" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## 🧠 Memory Management

### Store Memories
```bash
# Store user preference memory
curl -X POST "http://localhost:8003/memory" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "persona_id": "financial_advisor",
    "content": "User prefers conservative investment strategies",
    "content_type": "preference",
    "metadata": {
      "tags": ["investment", "conservative"],
      "importance": 8,
      "topic": "investment_strategy"
    }
  }'

# Store factual memory
curl -X POST "http://localhost:8003/memory" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "persona_id": "financial_advisor",
    "content": "User has monthly income of $5000",
    "content_type": "fact",
    "metadata": {
      "tags": ["income", "budget"],
      "importance": 9,
      "topic": "financial_profile"
    }
  }'

# Store interaction
curl -X POST "http://localhost:8003/memory/interaction" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "persona_id": "financial_advisor",
    "user_message": "What is the best investment strategy?",
    "agent_response": "Based on your conservative preference, I recommend a balanced portfolio.",
    "context": {
      "session_id": "session_001",
      "conversation_turn": 1,
      "domain": "finance"
    }
  }'
```

### Retrieve Memories
```bash
# Get all memories for a persona
curl -X GET "http://localhost:8003/memory?persona=financial_advisor&limit=10" \
  -H "Authorization: Bearer $API_KEY"

# Get memories for specific user
curl -X GET "http://localhost:8003/memory?persona=financial_advisor&user_id=user123&limit=5" \
  -H "Authorization: Bearer $API_KEY"

# Search memories
curl -X GET "http://localhost:8003/memory/search?query=investment%20strategy&limit=5" \
  -H "Authorization: Bearer $API_KEY"

# Get persona summary
curl -X GET "http://localhost:8003/memory/persona/financial_advisor/summary" \
  -H "Authorization: Bearer $API_KEY"
```

---

## 🌐 Multilingual Services (Karthikeya)

### Generate Reports
```bash
# Generate Edumentor report in Hindi
curl -X POST "http://localhost:5000/generate-report" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "student_123",
    "context": "edumentor",
    "language": "hi",
    "user_data": {
      "average_score": 75,
      "completed_lessons": 12,
      "streak_days": 5,
      "subject_area": "Mathematics"
    },
    "historical_data": {
      "previous_scores": [70, 75, 80, 72],
      "trend": "improving"
    }
  }'

# Generate Wellness report in English
curl -X POST "http://localhost:5000/generate-report" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "wellness_user_456",
    "context": "wellness",
    "language": "en",
    "user_data": {
      "spending_ratio": 0.8,
      "stress_level": 40,
      "savings_rate": 8,
      "score": 75
    },
    "historical_data": {
      "spending_trend": "decreasing",
      "stress_trend": "low"
    }
  }'
```

### Generate Nudges
```bash
# Generate Edumentor nudges
curl -X POST "http://localhost:5000/generate-nudge" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "nudge_test_student",
    "context": "edumentor",
    "language": "en",
    "user_data": {
      "average_score": 45,
      "missed_quizzes": 3,
      "engagement_score": 30,
      "streak_days": 0,
      "subject_area": "Mathematics"
    },
    "historical_data": {
      "previous_scores": [50, 48, 45, 42],
      "trend": "declining"
    },
    "preferences": {
      "notification_time": "evening",
      "tone_preference": "encouraging"
    }
  }'

# Generate Wellness nudges
curl -X POST "http://localhost:5000/generate-nudge" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "nudge_test_wellness",
    "context": "wellness",
    "language": "hi",
    "user_data": {
      "spending_ratio": 1.2,
      "stress_level": 80,
      "savings_rate": 3,
      "score": 40
    },
    "historical_data": {
      "spending_trend": "increasing",
      "stress_trend": "high"
    },
    "preferences": {
      "notification_frequency": "daily",
      "language_preference": "hi"
    }
  }'
```

---

## 🔊 Text-to-Speech Services

### Generate Audio
```bash
# Generate TTS
curl -X POST "http://localhost:8006/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to Gurukul Learning Platform!",
    "voice": "en-US-Neural2-F",
    "speed": 1.0
  }'

# Generate streaming TTS
curl -X POST "http://localhost:8006/api/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a streaming audio response.",
    "voice": "en-US-Neural2-M"
  }'

# Dedicated chatbot TTS
curl -X POST "http://localhost:8007/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from the dedicated chatbot service!",
    "voice": "en-US-Neural2-F"
  }'

# Streaming TTS from dedicated service
curl -X POST "http://localhost:8007/tts/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Streaming audio from dedicated service",
    "voice": "en-US-Neural2-M"
  }'
```

### Get Audio Files
```bash
# Get specific audio file
curl -X GET "http://localhost:8006/api/audio/filename.mp3"

# List all audio files
curl -X GET "http://localhost:8006/api/list-audio-files"

# Get audio files from subject generation
curl -X GET "http://localhost:8005/api/audio-files"

# Get specific audio from subject generation
curl -X GET "http://localhost:8005/api/audio/filename.mp3"
```

---

## 📄 Document Processing

### PDF Processing
```bash
# Process PDF
curl -X POST "http://localhost:8000/process-pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Summarize PDF
curl -X GET "http://localhost:8000/summarize-pdf?filename=document.pdf"

# Process PDF in API data service
curl -X POST "http://localhost:8001/process-pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Image Processing
```bash
# Process image
curl -X POST "http://localhost:8000/process-img" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Summarize image
curl -X GET "http://localhost:8000/summarize-img?filename=image.jpg"

# Process image in API data service
curl -X POST "http://localhost:8001/process-img" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

---

## 🎥 Video Services

### Video Processing
```bash
# Test video generation
curl -X POST "http://localhost:8000/test-generate-video" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to Gurukul Learning Platform",
    "duration": 10
  }'

# Receive video
curl -X POST "http://localhost:8000/receive-video" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@video.mp4"

# Get video info
curl -X GET "http://localhost:8000/videos/video_id_here/info"

# Get all videos
curl -X GET "http://localhost:8000/videos"

# Get specific video
curl -X GET "http://localhost:8000/videos/video_id_here"
```

---

## 🤖 Agent Simulation

### Subject Generation Agent
```bash
# Start agent simulation
curl -X POST "http://localhost:8005/start_agent_simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "educational",
    "user_id": "user123"
  }'

# Stop agent simulation
curl -X POST "http://localhost:8005/stop_agent_simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_123"
  }'

# Reset agent simulation
curl -X POST "http://localhost:8005/reset_agent_simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_123"
  }'

# Send agent message
curl -X POST "http://localhost:8005/agent_message" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_123",
    "message": "Hello agent!",
    "user_id": "user123"
  }'

# Get agent output
curl -X GET "http://localhost:8005/get_agent_output?agent_id=agent_123"

# Get agent logs
curl -X GET "http://localhost:8005/agent_logs?agent_id=agent_123"
```

---

## 👤 User Analytics

### User Progress
```bash
# Get user progress
curl -X GET "http://localhost:8000/user-progress/user123"

# Get user analytics
curl -X GET "http://localhost:8000/user-analytics/user123"

# Trigger intervention
curl -X POST "http://localhost:8000/trigger-intervention/user123" \
  -H "Content-Type: application/json" \
  -d '{
    "intervention_type": "motivational",
    "reason": "declining_performance"
  }'
```

---

## 🔍 Vision Services

### Proxy Vision
```bash
# Process image with vision
curl -X POST "http://localhost:8000/proxy/vision" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "prompt=Describe this image"
```

---

## 📊 Data Export

### Lesson Export
```bash
# Export lessons
curl -X POST "http://localhost:8005/lessons/export" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Mathematics",
    "format": "pdf",
    "include_audio": true
  }'
```

### Forward Data
```bash
# Forward data to external service
curl -X POST "http://localhost:8005/forward_data" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "lesson_content",
    "target_service": "external_lms",
    "user_id": "user123"
  }'

# Send lesson to external
curl -X POST "http://localhost:8005/send_lesson_to_external" \
  -H "Content-Type: application/json" \
  -d '{
    "lesson_id": "lesson_123",
    "external_url": "https://external-lms.com/api/lessons"
  }'
```

---

## 🔧 Utility Endpoints

### Check External Server
```bash
# Check external server status
curl -X GET "http://localhost:8005/check_external_server?url=https://api.example.com"
```

### Static Files
```bash
# Get static file
curl -X GET "http://localhost:8007/static/filename.ext"
```

### Stream Files
```bash
# Get stream file
curl -X GET "http://localhost:8000/api/stream/filename.ext"

# Get audio stream
curl -X GET "http://localhost:8000/api/audio/filename.mp3"
```

---

## 🧪 Testing Scripts

### Run Memory Management Tests
```bash
cd Backend/memory_management
bash curl_examples.sh
```

### Run Financial Simulator Tests
```bash
cd Backend/Financial_simulator
bash curl_examples.sh
```

### Run Karthikeya Tests
```bash
cd Backend/Karthikeya
bash curl_test.sh
```

### Run MVD Flow Test

#### Using Python Test Script:
```bash
cd Backend/Base_backend
python test_mvd_endpoints.py
```

#### Using Windows Batch File:
```cmd
cd Backend\Base_backend
test_mvd.bat
```

#### Manual Testing:
```bash
# Test the complete MVD flow
bash -c '
echo "=== Testing MVD Flow ==="
curl -X GET "http://localhost:8000/dashboard/user123" | jq "."
curl -X GET "http://localhost:8000/syllabus/math_001" | jq "."
curl -X GET "http://localhost:8000/lesson/lesson_001" | jq "."
curl -X GET "http://localhost:8000/quiz/lesson_001" | jq "."
curl -X POST "http://localhost:8000/lesson-complete" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"user123\", \"lesson_id\": \"lesson_001\", \"quiz_score\": 85.5, \"time_spent\": 45}" | jq "."
echo "=== MVD Flow Test Complete ==="
'
```

---

## 📝 Notes

1. **Authentication**: Most endpoints require proper authentication headers
2. **File Uploads**: Use `-F` for multipart form data uploads
3. **JSON Formatting**: Use `jq` for better JSON formatting: `curl ... | jq .`
4. **Error Handling**: Check HTTP status codes for error responses
5. **Rate Limiting**: Be mindful of API rate limits
6. **File Paths**: Replace `filename.ext` with actual filenames
7. **IDs**: Replace `user123`, `agent_123`, etc. with actual IDs

## 🚨 Troubleshooting

### Common Issues
- **Port not available**: Check if services are running on correct ports
- **Authentication errors**: Verify API keys and JWT tokens
- **File not found**: Ensure files exist in the specified paths
- **Database connection**: Check MongoDB connection for memory management
- **CORS errors**: Verify CORS configuration for frontend integration

### Debug Commands
```bash
# Check if ports are listening
netstat -an | findstr :8000
netstat -an | findstr :8001
netstat -an | findstr :8002
netstat -an | findstr :8003
netstat -an | findstr :8004
netstat -an | findstr :8005
netstat -an | findstr :5000

# Test basic connectivity
curl -v http://localhost:8000/health
curl -v http://localhost:8001/health
curl -v http://localhost:8002/health
```

---

**Happy Testing! 🎉**
