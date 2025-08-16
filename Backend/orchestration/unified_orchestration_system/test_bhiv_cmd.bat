@echo off
echo ========================================
echo 🧪 BHIV Core CMD Testing Suite
echo ========================================

set API_URL=http://localhost:8010

echo.
echo 🔍 Testing BHIV Core API...
echo API URL: %API_URL%
echo.

echo ========================================
echo Test 1: Health Check
echo ========================================
curl -s %API_URL%/health
echo.
echo.

echo ========================================
echo Test 2: Root Endpoint
echo ========================================
curl -s %API_URL%/
echo.
echo.

echo ========================================
echo Test 3: System Statistics
echo ========================================
curl -s %API_URL%/api/v1/bhiv-core/stats
echo.
echo.

echo ========================================
echo Test 4: Query Classification
echo ========================================
echo Testing Educational Query...
curl -s -X POST "%API_URL%/api/v1/bhiv-core/classify" ^
  -H "Content-Type: application/json" ^
  -d @test_data/classify_educational.json
echo.
echo.

echo Testing Wellness Query...
curl -s -X POST "%API_URL%/api/v1/bhiv-core/classify" ^
  -H "Content-Type: application/json" ^
  -d @test_data/classify_wellness.json
echo.
echo.

echo Testing Spiritual Query (using inline JSON)...
echo {"user_id": "test_user_003", "query": "What do the Vedas say about life purpose?", "lang": "en-IN"} > temp_spiritual.json
curl -s -X POST "%API_URL%/api/v1/bhiv-core/classify" ^
  -H "Content-Type: application/json" ^
  -d @temp_spiritual.json
del temp_spiritual.json
echo.
echo.

echo ========================================
echo Test 5: Quick Query Processing
echo ========================================
echo Testing Simple Educational Query...
curl -s -X POST "%API_URL%/api/v1/bhiv-core/quick-query" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": \"student_123\", \"query\": \"Explain machine learning basics\", \"voice_enabled\": false, \"lang\": \"en-IN\"}"
echo.
echo.

echo Testing Voice-Enabled Query...
curl -s -X POST "%API_URL%/api/v1/bhiv-core/quick-query" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": \"student_456\", \"query\": \"What is photosynthesis?\", \"voice_enabled\": true, \"lang\": \"en-IN\"}"
echo.
echo.

echo ========================================
echo Test 6: Full BHIV Processing
echo ========================================
echo Testing Complete BHIV Request...
curl -s -X POST "%API_URL%/api/v1/bhiv-core/process" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\": \"test-session-cmd\", \"timestamp\": \"2025-08-13T16:00:00Z\", \"user_info\": {\"user_id\": \"cmd_test_user\", \"auth_token\": \"bhiv-core-key\", \"lang\": \"en-IN\", \"persona\": \"student\", \"permissions\": [\"read\", \"write\"]}, \"input\": {\"raw_text\": \"Explain quantum computing applications\", \"voice_enabled\": false, \"mode\": \"chat\"}, \"knowledgebase_query\": {\"use_vector\": true, \"top_k\": 3}, \"llm_request\": {\"model\": \"gemini\", \"max_tokens\": 512}}"
echo.
echo.

echo ========================================
echo 🎯 Testing Complete!
echo ========================================
echo.
echo 💡 To see formatted output, visit: %API_URL%/docs
echo 📊 For detailed testing, run: python test_api_client.py
echo.
pause
