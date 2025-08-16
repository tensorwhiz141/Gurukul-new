@echo off
echo ========================================
echo    🤖 Dedicated Chatbot Service
echo ========================================
echo.
echo Starting standalone chatbot service on port 8001...
echo This service is separated from other backend services
echo to avoid conflicts and ensure reliable chat functionality.
echo.

REM Check if port 8001 is already in use
echo 🔍 Checking if port 8001 is available...
netstat -ano | findstr :8001 >nul
if %errorlevel% == 0 (
    echo ⚠️  Port 8001 is already in use. Attempting to free it...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
        echo Killing process %%a...
        taskkill /PID %%a /F >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

REM Navigate to the chatbot service directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn, pymongo, pydantic" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Installing required packages...
    pip install fastapi uvicorn pymongo pydantic python-dotenv requests
)

REM Check if .env file exists in Base_backend
if not exist "..\Base_backend\.env" (
    echo ❌ .env file not found in Base_backend directory
    echo Please ensure the .env file exists with GROQ_API_KEY and MONGO_URI
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Dedicated Chatbot Service...
echo.
echo 📍 Service will be available at:
echo    - Main endpoint: http://localhost:8001/chatbot
echo    - Health check:  http://localhost:8001/health
echo    - Test endpoint: http://localhost:8001/test
echo    - API docs:      http://localhost:8001/docs
echo.
echo 💡 This service handles:
echo    - POST /chatpost - Receive chat messages
echo    - GET /chatbot   - Get AI responses
echo    - GET /health    - Service health check
echo    - GET /chat-history - Get user chat history
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start the chatbot service
python chatbot_api.py

echo.
echo 🛑 Chatbot service stopped.
pause
