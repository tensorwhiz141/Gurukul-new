@echo off
echo ========================================
echo    Gurukul Unified Agent Mind
echo    Starting All Backend Services
echo ========================================
echo.

REM Set the base directory
set BASE_DIR=%~dp0

echo 🚀 Starting all backend services with orchestration...
echo.

REM Check and install critical dependencies
echo 📦 Checking critical dependencies...
python -c "import fastapi" 2>nul || (
    echo ⚠️  Installing missing fastapi dependency...
    pip install fastapi>=0.95.0 uvicorn>=0.20.0 --quiet
)

python -c "import langchain_groq" 2>nul || (
    echo ⚠️  Installing missing langchain_groq dependency...
    pip install langchain>=0.1.0 langchain-groq>=0.1.0 langgraph>=0.0.30 --quiet
)

python -c "import prophet" 2>nul || (
    echo ⚠️  Installing missing prophet dependency...
    pip install prophet>=1.1.4 --quiet
)

echo ✅ Dependencies checked
echo.

REM Start Base Backend with Orchestration (Port 8000) - Main API
echo 🏠 Starting Base Backend with Orchestration on port 8000...
start "Base Backend (Main API)" cmd /k "cd /d %BASE_DIR%Base_backend && python api.py"
timeout /t 4 /nobreak >nul

REM Start API Data Service (Port 8001)
echo 🤖 Starting API Data Service on port 8001...
start "API Data Service" cmd /k "cd /d %BASE_DIR%api_data && python api.py"
timeout /t 3 /nobreak >nul

REM Start Financial Simulator (Port 8002)
echo 💰 Starting Financial Simulator on port 8002...
start "Financial Simulator" cmd /k "cd /d %BASE_DIR%Financial_simulator\Financial_simulator && python langgraph_api.py"
timeout /t 3 /nobreak >nul

REM Start Memory Management API (Port 8003)
echo 📝 Starting Memory Management API on port 8003...
start "Memory Management API" cmd /k "cd /d %BASE_DIR%memory_management && python run_server.py"
timeout /t 3 /nobreak >nul

REM Start Akash Service (Port 8004)
echo 🧠 Starting Akash Service on port 8004...
start "Akash Service" cmd /k "cd /d %BASE_DIR%akash && python main.py"
timeout /t 3 /nobreak >nul

REM Start Subject Generation Service (Port 8005)
echo 📖 Starting Subject Generation Service on port 8005...
start "Subject Generation" cmd /k "cd /d %BASE_DIR%subject_generation && python app.py"
timeout /t 3 /nobreak >nul

REM Start Wellness API with Advanced Forecasting (Port 8006)
echo 🧘🔮 Starting Wellness API with Advanced Forecasting on port 8006...
start "Wellness API + Forecasting" cmd /k "cd /d %BASE_DIR%orchestration\unified_orchestration_system && python simple_api.py --port 8006"
timeout /t 4 /nobreak >nul

REM Start TTS Service (Port 8007)
echo 🔊 Starting TTS Service on port 8007...
start "TTS Service" cmd /k "cd /d %BASE_DIR%tts_service && python tts.py"
timeout /t 3 /nobreak >nul

echo.
echo ✅ All 8 backend services are starting...
echo.
echo 🌐 Service URLs:
echo    Base Backend (Main API):     http://localhost:8000/health
echo    API Data Service:            http://localhost:8001/health
echo    Financial Simulator:         http://localhost:8002/health
echo    Memory Management API:       http://localhost:8003/memory/health
echo    Akash Service:               http://localhost:8004/health
echo    Subject Generation:          http://localhost:8005/health
echo    Wellness API + Forecasting:  http://localhost:8006/
echo    TTS Service:                 http://localhost:8007/api/health
echo.
echo 📋 Next Steps:
echo    1. Wait 20-30 seconds for all services to start
echo    2. Check the service URLs above to verify they're running
echo    3. Start the frontend: cd "new frontend" && start_frontend.bat
echo    4. Open http://localhost:3000 or http://localhost:5174 in your browser
echo.
echo 🔧 To stop all services: Close all the opened terminal windows
echo.
pause
