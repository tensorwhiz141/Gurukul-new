@echo off
echo ========================================
echo   GURUKUL LESSON GENERATION SERVICES
echo ========================================
echo.
echo Starting lesson generation services...
echo.

REM Set the base directory
set BASE_DIR=%~dp0
cd /d "%BASE_DIR%"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo [1/4] Starting TTS Service (Port 8007)...
echo.

REM Start TTS service on port 8007
start "TTS Service" cmd /k "cd /d "%BASE_DIR%tts_service" && python tts.py"

REM Wait a moment for the TTS service to start
timeout /t 3 /nobreak >nul

echo [2/4] Starting Orchestration System with Forecasting (Port 8002)...
echo.

REM Install forecasting dependencies if needed
echo Installing forecasting dependencies...
pip install prophet>=1.1.4 statsmodels>=0.14.0 scikit-learn>=1.3.0 --quiet >nul 2>&1

REM Start orchestration system on port 8002
start "Orchestration System" cmd /k "cd /d "%BASE_DIR%orchestration\unified_orchestration_system" && python simple_api.py --port 8002 --host 0.0.0.0"

REM Wait a moment for the orchestration system to start
timeout /t 5 /nobreak >nul

echo [3/4] Starting Subject Generation System (Port 8000)...
echo.

REM Start subject generation system on port 8000
start "Subject Generation" cmd /k "cd /d "%BASE_DIR%subject_generation" && python app.py"

REM Wait a moment for the subject generation system to start
timeout /t 5 /nobreak >nul

echo [4/4] Testing service connectivity...
echo.

REM Test TTS service
echo Testing TTS Service (http://localhost:8007)...
curl -s "http://localhost:8007/api/health" >nul 2>&1
if errorlevel 1 (
    echo WARNING: TTS Service may not be running properly
) else (
    echo ✓ TTS Service is running
)

REM Test orchestration system
echo Testing Orchestration System (http://localhost:8002)...
curl -s "http://localhost:8002/" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Orchestration System may not be running properly
) else (
    echo ✓ Orchestration System is running
)

REM Test forecasting functionality
echo Testing Advanced Forecasting (http://localhost:8002/forecast/status)...
curl -s "http://localhost:8002/forecast/status" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Forecasting may not be available
) else (
    echo ✓ Advanced Forecasting is enabled
)

REM Test subject generation system
echo Testing Subject Generation System (http://localhost:8000)...
curl -s "http://localhost:8000/" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Subject Generation System may not be running properly
) else (
    echo ✓ Subject Generation System is running
)

echo.
echo ========================================
echo   SERVICES STARTED SUCCESSFULLY
echo ========================================
echo.
echo Service URLs:
echo   • TTS Service:          http://localhost:8007/api/health
echo   • Orchestration System: http://localhost:8002
echo   • Advanced Forecasting: http://localhost:8002/forecast/status
echo   • Subject Generation:   http://localhost:8000
echo   • API Documentation:    http://localhost:8000/docs
echo.
echo Available Endpoints:
echo   • Wikipedia Only:       GET /generate_lesson?subject=maths&topic=algebra&include_wikipedia=true&use_knowledge_store=false
echo   • Knowledge Store Only: GET /generate_lesson?subject=maths&topic=algebra&include_wikipedia=false&use_knowledge_store=true
echo   • Combined Mode:        GET /generate_lesson?subject=maths&topic=algebra&include_wikipedia=true&use_knowledge_store=true
echo.
echo Press any key to run a test lesson generation...
pause >nul

echo.
echo ========================================
echo   TESTING LESSON GENERATION
echo ========================================
echo.

echo [TEST 1] Wikipedia Only Mode...
curl -s "http://localhost:8000/generate_lesson?subject=maths&topic=algebra&include_wikipedia=true&use_knowledge_store=false" > test_wikipedia.json
if errorlevel 1 (
    echo ❌ Wikipedia test failed
) else (
    echo ✓ Wikipedia test completed - check test_wikipedia.json
)

echo.
echo [TEST 2] Knowledge Store Only Mode...
curl -s "http://localhost:8000/generate_lesson?subject=maths&topic=algebra&include_wikipedia=false&use_knowledge_store=true" > test_knowledge_store.json
if errorlevel 1 (
    echo ❌ Knowledge Store test failed
) else (
    echo ✓ Knowledge Store test completed - check test_knowledge_store.json
)

echo.
echo [TEST 3] Combined Mode...
curl -s "http://localhost:8000/generate_lesson?subject=maths&topic=algebra&include_wikipedia=true&use_knowledge_store=true" > test_combined.json
if errorlevel 1 (
    echo ❌ Combined test failed
) else (
    echo ✓ Combined test completed - check test_combined.json
)

echo.
echo [TEST 4] Advanced Forecasting...
curl -s "http://localhost:8002/forecast/status" > test_forecasting.json
if errorlevel 1 (
    echo ❌ Forecasting test failed
) else (
    echo ✓ Forecasting test completed - check test_forecasting.json
)

echo.
echo ========================================
echo   TESTING COMPLETED
echo ========================================
echo.
echo Test results saved to:
echo   • test_wikipedia.json
echo   • test_knowledge_store.json
echo   • test_combined.json
echo   • test_forecasting.json
echo.
echo Services are running in separate windows.
echo Close those windows to stop the services.
echo.
echo Press any key to exit...
pause >nul
