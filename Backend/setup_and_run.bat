@echo off
echo ========================================
echo    Gurukul Unified Agent Mind
echo    Automated Setup and Run Script
echo ========================================
echo.

REM Set the base directory
set BASE_DIR=%~dp0

echo 🔧 Setting up environment...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

echo ✅ Python and Node.js are installed
echo.

REM Create .env files if they don't exist
echo 📝 Creating environment files...

REM Memory Management API .env
if not exist "%BASE_DIR%memory_management\.env" (
    echo Creating memory_management/.env...
    (
        echo MONGODB_URL=mongodb://localhost:27017/
        echo MEMORY_DB_NAME=gurukul
        echo MEMORY_API_KEYS=memory_api_key_dev:development_user,memory_api_key_prod:production_user
        echo MEMORY_API_HOST=0.0.0.0
        echo MEMORY_API_PORT=8003
        echo MEMORY_RATE_LIMIT_REQUESTS=1000
        echo MEMORY_RATE_LIMIT_WINDOW=3600
    ) > "%BASE_DIR%memory_management\.env"
)

<<<<<<< HEAD
REM Base Backend .env
if not exist "%BASE_DIR%Base_backend\.env" (
    echo Creating Base_backend/.env...
    (
        echo GROQ_API_KEY=your_groq_api_key_here
        echo GEMINI_API_KEY=your_gemini_api_key_here
        echo MONGODB_URL=mongodb://localhost:27017/
        echo CHROMA_PERSIST_DIRECTORY=knowledge_store
    ) > "%BASE_DIR%Base_backend\.env"
)

REM Dedicated Chatbot Service .env
if not exist "%BASE_DIR%dedicated_chatbot_service\.env" (
    echo Creating dedicated_chatbot_service/.env...
    (
        echo GROQ_API_KEY=your_groq_api_key_here
        echo MONGODB_URL=mongodb://localhost:27017/
        echo CHATBOT_DB_NAME=gurukul_chatbot
        echo CHATBOT_API_HOST=0.0.0.0
        echo CHATBOT_API_PORT=8001
    ) > "%BASE_DIR%dedicated_chatbot_service\.env"
)

REM Financial Simulator .env
if not exist "%BASE_DIR%Financial_simulator\.env" (
=======
REM API Data Service .env
if not exist "%BASE_DIR%api_data\.env" (
    echo Creating api_data/.env...
    (
        echo GROQ_API_KEY=your_groq_api_key_here
        echo MONGODB_URL=mongodb://localhost:27017/
    ) > "%BASE_DIR%api_data\.env"
)

REM Financial Simulator .env
if not exist "%BASE_DIR%Financial_simulator\Financial_simulator\.env" (
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
    echo Creating Financial_simulator/.env...
    (
        echo OPENAI_API_KEY=your_openai_key_here
        echo MONGODB_URL=mongodb://localhost:27017/
<<<<<<< HEAD
    ) > "%BASE_DIR%Financial_simulator\.env"
)

REM Akash Service .env
if not exist "%BASE_DIR%akash\.env" (
    echo Creating akash/.env...
    (
        echo GROQ_API_KEY=your_groq_api_key_here
        echo MONGODB_URL=mongodb://localhost:27017/
    ) > "%BASE_DIR%akash\.env"
)

REM Subject Generation .env
if not exist "%BASE_DIR%subject_generation\.env" (
    echo Creating subject_generation/.env...
    (
        echo GROQ_API_KEY=your_groq_api_key_here
        echo OPENAI_API_KEY=your_openai_key_here
        echo CHROMA_PERSIST_DIRECTORY=knowledge_store
    ) > "%BASE_DIR%subject_generation\.env"
)

REM Wellness API (Orchestration) .env
if not exist "%BASE_DIR%orchestration\unified_orchestration_system\.env" (
    echo Creating orchestration/unified_orchestration_system/.env...
    (
        echo GEMINI_API_KEY=your_gemini_api_key_here
        echo MONGODB_URL=mongodb://localhost:27017/
    ) > "%BASE_DIR%orchestration\unified_orchestration_system\.env"
=======
    ) > "%BASE_DIR%Financial_simulator\Financial_simulator\.env"
)

REM Lesson Generator .env
if not exist "%BASE_DIR%pipline-24-master\.env" (
    echo Creating pipline-24-master/.env...
    (
        echo OPENAI_API_KEY=your_openai_key_here
        echo CHROMA_PERSIST_DIRECTORY=knowledge_store
    ) > "%BASE_DIR%pipline-24-master\.env"
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
)

echo ✅ Environment files created
echo.

echo ⚠️  IMPORTANT: Please update the API keys in the .env files:
echo    - Get OpenAI API key from: https://platform.openai.com/api-keys
echo    - Get Groq API key from: https://console.groq.com/keys
echo    - Update the .env files with your actual API keys
echo.

echo 📦 Installing Python dependencies...
echo.

REM Install dependencies for each service
<<<<<<< HEAD
echo Installing Base Backend dependencies...
cd /d "%BASE_DIR%Base_backend"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn python-dotenv pymongo requests groq langchain chromadb sentence-transformers >nul 2>&1
)

echo Installing Dedicated Chatbot Service dependencies...
cd /d "%BASE_DIR%dedicated_chatbot_service"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn python-dotenv pymongo requests groq >nul 2>&1
)

echo Installing Financial Simulator dependencies...
cd /d "%BASE_DIR%Financial_simulator"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn langgraph langchain openai python-dotenv agentops >nul 2>&1
)

echo Installing Memory Management API dependencies...
cd /d "%BASE_DIR%memory_management"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn pydantic pymongo python-dotenv python-dateutil structlog >nul 2>&1
)

echo Installing Akash Service dependencies...
cd /d "%BASE_DIR%akash"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn python-dotenv pymongo requests groq >nul 2>&1
)

echo Installing Subject Generation dependencies...
cd /d "%BASE_DIR%subject_generation"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn python-dotenv requests torch transformers gtts groq openai langchain chromadb >nul 2>&1
)

echo Installing Wellness API (Orchestration) dependencies...
cd /d "%BASE_DIR%orchestration\unified_orchestration_system"
if exist requirements.txt (
    pip install -r requirements.txt >nul 2>&1
) else (
    pip install fastapi uvicorn python-dotenv google-generativeai >nul 2>&1
)
=======
echo Installing Memory Management API dependencies...
cd /d "%BASE_DIR%memory_management"
pip install fastapi uvicorn pydantic pymongo python-dotenv python-dateutil structlog >nul 2>&1

echo Installing API Data Service dependencies...
cd /d "%BASE_DIR%api_data"
pip install fastapi uvicorn python-dotenv pymongo requests groq langchain chromadb sentence-transformers >nul 2>&1

echo Installing Financial Simulator dependencies...
cd /d "%BASE_DIR%Financial_simulator\Financial_simulator"
pip install fastapi uvicorn langgraph langchain openai python-dotenv agentops >nul 2>&1

echo Installing Lesson Generator dependencies...
cd /d "%BASE_DIR%pipline-24-master"
pip install fastapi uvicorn python-dotenv requests torch transformers gtts >nul 2>&1
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d

echo ✅ Python dependencies installed
echo.

echo 📦 Installing Frontend dependencies...
<<<<<<< HEAD
cd /d "%BASE_DIR%..\new frontend"
=======
cd /d "%BASE_DIR%..\gurukul_frontend-main"
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
if exist package.json (
    npm install >nul 2>&1
    echo ✅ Frontend dependencies installed
) else (
    echo ⚠️  Frontend directory not found at expected location
<<<<<<< HEAD
    echo Please make sure "new frontend" directory exists in the parent directory
)
echo.

echo 🚀 Starting all 7 backend services with orchestration...
echo.

REM Start Base Backend with Orchestration (Port 8000) - Main API
echo 🏗️ Starting Base Backend with Orchestration on port 8000...
start "Base Backend (Main API)" cmd /k "cd /d %BASE_DIR%Base_backend && echo Starting Base Backend... && python api.py"
timeout /t 4 /nobreak >nul

REM Start Dedicated Chatbot Service (Port 8001)
echo 🤖 Starting Dedicated Chatbot Service on port 8001...
start "Dedicated Chatbot Service" cmd /k "cd /d %BASE_DIR%dedicated_chatbot_service && echo Starting Dedicated Chatbot Service... && python chatbot_api.py"
timeout /t 3 /nobreak >nul

REM Start Financial Simulator (Port 8002)
echo 💰 Starting Financial Simulator on port 8002...
start "Financial Simulator" cmd /k "cd /d %BASE_DIR%Financial_simulator && echo Starting Financial Simulator... && python langgraph_api.py"
timeout /t 3 /nobreak >nul

=======
    echo Please make sure gurukul_frontend-main is in the parent directory
)
echo.

echo 🚀 Starting all services...
echo.

>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
REM Start Memory Management API (Port 8003)
echo 📝 Starting Memory Management API on port 8003...
start "Memory Management API" cmd /k "cd /d %BASE_DIR%memory_management && echo Starting Memory Management API... && python run_server.py"
timeout /t 3 /nobreak >nul

<<<<<<< HEAD
REM Start Akash Service (Port 8004)
echo 🧠 Starting Akash Service on port 8004...
start "Akash Service" cmd /k "cd /d %BASE_DIR%akash && echo Starting Akash Service... && python main.py"
timeout /t 3 /nobreak >nul

REM Start Subject Generation Service (Port 8005)
echo 📖 Starting Subject Generation Service on port 8005...
start "Subject Generation" cmd /k "cd /d %BASE_DIR%subject_generation && echo Starting Subject Generation... && python app.py"
timeout /t 3 /nobreak >nul

REM Start Wellness API (Unified Orchestration System) (Port 8006)
echo 🧘 Starting Wellness API (Unified Orchestration) on port 8006...
start "Wellness API" cmd /k "cd /d %BASE_DIR%orchestration\unified_orchestration_system && echo Starting Wellness API... && python simple_api.py --port 8006"
timeout /t 4 /nobreak >nul

REM Start Frontend (Port 3000) - Vite project
echo 🌐 Starting Frontend on port 3000...
cd /d "%BASE_DIR%..\new frontend"
if exist package.json (
    start "Gurukul Frontend" cmd /k "echo Starting Frontend (Vite)... && npm start"
=======
REM Start API Data Service (Port 8001)
echo 🤖 Starting API Data Service on port 8001...
start "API Data Service" cmd /k "cd /d %BASE_DIR%api_data && echo Starting API Data Service... && python api.py"
timeout /t 3 /nobreak >nul

REM Start Financial Simulator (Port 8002)
echo 💰 Starting Financial Simulator on port 8002...
start "Financial Simulator" cmd /k "cd /d %BASE_DIR%Financial_simulator\Financial_simulator && echo Starting Financial Simulator... && python langgraph_api.py"
timeout /t 3 /nobreak >nul

REM Start Lesson Generator (Port 8000)
echo 📚 Starting Lesson Generator on port 8000...
start "Lesson Generator" cmd /k "cd /d %BASE_DIR%pipline-24-master && echo Starting Lesson Generator... && python app.py"
timeout /t 3 /nobreak >nul

REM Start Frontend (Port 3000) - CORRECTED: This is a Vite project
echo 🌐 Starting Frontend on port 3000...
cd /d "%BASE_DIR%..\gurukul_frontend-main"
if exist package.json (
    start "Gurukul Frontend" cmd /k "echo Starting Frontend (Vite)... && npm run dev"
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
) else (
    echo ⚠️  Could not start frontend - package.json not found
)

echo.
<<<<<<< HEAD
echo ✅ All 7 backend services are starting...
echo.
echo 🌐 Service URLs (wait 20-30 seconds for all to start):
echo    Base Backend (Main API):     http://localhost:8000/health
echo    Dedicated Chatbot Service:   http://localhost:8001/health
echo    Financial Simulator:         http://localhost:8002/health
echo    Memory Management API:       http://localhost:8003/memory/health
echo    Akash Service:               http://localhost:8004/health
echo    Subject Generation:          http://localhost:8005/health
echo    Wellness API (Orchestration): http://localhost:8006/
echo    Frontend:                    http://localhost:3000
echo.
echo 🚀 Orchestration Integration Status:
echo    Integration Status:          http://localhost:8000/integration-status
echo    Enhanced Lesson Generation:  http://localhost:8000/lessons/enhanced
echo    User Progress Tracking:      http://localhost:8000/user-progress/{user_id}
echo.
echo 📋 Next Steps:
echo    1. Wait 20-30 seconds for all services to start
echo    2. Check the service URLs above to verify they're running
echo    3. Verify orchestration integration status
echo    4. Open http://localhost:3000 in your browser
echo    5. Look for "AI Enhanced" indicator in Subject Explorer
echo    6. Test WellnessBot in Agent Simulator (should show "API Connected")
echo    7. Create an account and start using Gurukul!
echo.
echo 🔧 To stop all services: Close all the opened terminal windows
echo.
echo 💡 Troubleshooting:
echo    - If orchestration features don't work, add GEMINI_API_KEY to Base_backend/.env
echo    - Check Backend/Base_backend/ORCHESTRATION_INTEGRATION_GUIDE.md for setup
echo    - Run test: cd Backend/Base_backend && python test_orchestration_integration.py
echo    - For WellnessBot: Ensure GEMINI_API_KEY is set in Backend/orchestration/unified_orchestration_system/.env
echo    - WellnessBot endpoints: http://localhost:8006/wellness, http://localhost:8006/docs
echo.
=======
echo ✅ All services are starting...
echo.
echo 🌐 Service URLs (wait 15-30 seconds for all to start):
echo    Memory Management API: http://localhost:8003/memory/health
echo    API Data Service:      http://localhost:8001/health
echo    Financial Simulator:   http://localhost:8002/
echo    Lesson Generator:      http://localhost:8000/
echo    Frontend:              http://localhost:3000
echo.
echo 📋 Next Steps:
echo    1. Wait 15-30 seconds for all services to start
echo    2. Check the service URLs above to verify they're running
echo    3. Open http://localhost:3000 in your browser
echo    4. Create an account and start using Gurukul!
echo.
echo 🔧 To stop all services: Close all the opened terminal windows
echo.
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
echo 📖 For detailed instructions, see: RUN_ALL_STEP_BY_STEP.md
echo.

REM Wait and then check service health
<<<<<<< HEAD
echo Waiting 25 seconds before checking service health...
timeout /t 25 /nobreak >nul
=======
echo Waiting 20 seconds before checking service health...
timeout /t 20 /nobreak >nul
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d

echo.
echo 🔍 Checking service health...
echo ================================

<<<<<<< HEAD
curl -s http://localhost:8000/health >nul 2>&1 && echo ✅ Base Backend (Main API) - Running || echo ❌ Base Backend (Main API) - Not responding
curl -s http://localhost:8001/health >nul 2>&1 && echo ✅ Dedicated Chatbot Service - Running || echo ❌ Dedicated Chatbot Service - Not responding
curl -s http://localhost:8002/health >nul 2>&1 && echo ✅ Financial Simulator - Running || echo ❌ Financial Simulator - Not responding
curl -s http://localhost:8003/memory/health >nul 2>&1 && echo ✅ Memory Management API - Running || echo ❌ Memory Management API - Not responding
curl -s http://localhost:8004/health >nul 2>&1 && echo ✅ Akash Service - Running || echo ❌ Akash Service - Not responding
curl -s http://localhost:8005/health >nul 2>&1 && echo ✅ Subject Generation - Running || echo ❌ Subject Generation - Not responding
curl -s http://localhost:8006/ >nul 2>&1 && echo ✅ Wellness API (Orchestration) - Running || echo ❌ Wellness API (Orchestration) - Not responding
=======
curl -s http://localhost:8003/memory/health >nul 2>&1 && echo ✅ Memory Management API - Running || echo ❌ Memory Management API - Not responding
curl -s http://localhost:8001/health >nul 2>&1 && echo ✅ API Data Service - Running || echo ❌ API Data Service - Not responding
curl -s http://localhost:8002/ >nul 2>&1 && echo ✅ Financial Simulator - Running || echo ❌ Financial Simulator - Not responding
curl -s http://localhost:8000/ >nul 2>&1 && echo ✅ Lesson Generator - Running || echo ❌ Lesson Generator - Not responding
>>>>>>> 0b6eca9f34ddee548c038c4179425b546ca35b5d
curl -s http://localhost:3000 >nul 2>&1 && echo ✅ Frontend - Running || echo ❌ Frontend - Not responding

echo.
echo 🎉 Setup complete! Your Gurukul system should now be running.
echo 🌐 Open http://localhost:3000 in your browser to get started!
echo.
pause
