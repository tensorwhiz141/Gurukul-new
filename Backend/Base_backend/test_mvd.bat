@echo off
echo ========================================
echo Testing MVD Endpoints
echo ========================================

echo.
echo 1. Testing Dashboard Endpoint...
curl -X GET "http://localhost:8000/dashboard/user123" -H "Content-Type: application/json"

echo.
echo 2. Testing Syllabus Endpoint...
curl -X GET "http://localhost:8000/syllabus/math_001" -H "Content-Type: application/json"

echo.
echo 3. Testing Lesson Endpoint...
curl -X GET "http://localhost:8000/lesson/lesson_001" -H "Content-Type: application/json"

echo.
echo 4. Testing Quiz Endpoint...
curl -X GET "http://localhost:8000/quiz/lesson_001" -H "Content-Type: application/json"

echo.
echo 5. Testing Lesson Complete Endpoint...
curl -X POST "http://localhost:8000/lesson-complete" -H "Content-Type: application/json" -d "{\"user_id\": \"user123\", \"lesson_id\": \"lesson_001\", \"quiz_score\": 85.5, \"time_spent\": 45}"

echo.
echo 6. Testing Health Endpoint (for comparison)...
curl -X GET "http://localhost:8000/health"

echo.
echo ========================================
echo MVD Testing Complete!
echo ========================================
pause
