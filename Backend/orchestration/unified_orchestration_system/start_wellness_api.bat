@echo off
echo ====================================
echo    WELLNESS API STARTUP SCRIPT
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if we're in the right directory
if not exist "orchestration_api.py" (
    if not exist "simple_api.py" (
        echo ERROR: API files not found
        echo Please run this script from the unified_orchestration_system directory
        pause
        exit /b 1
    )
)

echo.
echo Checking requirements...

REM Install requirements if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating requirements...
pip install -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo.
    echo Creating .env file...
    echo # Gemini API Keys > .env
    echo GEMINI_API_KEY=your_primary_key_here >> .env
    echo GEMINI_API_KEY_BACKUP=your_backup_key_here >> .env
    echo. >> .env
    echo # Sub-agent URLs (optional) >> .env
    echo EMOTIONAL_WELLNESS_BOT_URL=http://localhost:8002 >> .env
    echo FINANCIAL_WELLNESS_BOT_URL=http://localhost:8003 >> .env
    echo TUTORBOT_URL=http://localhost:8001 >> .env
    echo QUIZBOT_URL=http://localhost:8004 >> .env
    
    echo.
    echo ====================================
    echo    CONFIGURATION REQUIRED
    echo ====================================
    echo.
    echo A .env file has been created.
    echo Please edit it and add your Gemini API keys:
    echo.
    echo 1. Get API keys from: https://makersuite.google.com/app/apikey
    echo 2. Edit .env file and replace 'your_primary_key_here' with your actual key
    echo 3. Run this script again
    echo.
    pause
    exit /b 0
)

echo.
echo ====================================
echo    STARTING WELLNESS API SERVER
echo ====================================
echo.
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Endpoints:
echo   POST /ask-wellness - Full wellness support
echo   POST /wellness - Simple wellness advice
echo   POST /ask-vedas - Spiritual wisdom
echo   POST /edumentor - Educational content
echo.
echo Press Ctrl+C to stop the server
echo ====================================
echo.

REM Try to start the full orchestration API first, then fallback to simple API
if exist "orchestration_api.py" (
    echo Starting full orchestration system...
    python orchestration_api.py
) else (
    echo Starting simple API...
    python simple_api.py
)

echo.
echo Server stopped.
pause
