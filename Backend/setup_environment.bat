@echo off
echo ========================================
echo   GURUKUL ENVIRONMENT SETUP
echo ========================================
echo.
echo Setting up environment variables and API keys...
echo.

REM Set the base directory
set BASE_DIR=%~dp0

echo 📋 Creating environment files for all services...
echo.

REM Create .env file for Base Backend
echo [1/8] Setting up Base Backend environment...
if not exist "%BASE_DIR%Base_backend\.env" (
    echo # Base Backend Environment Variables > "%BASE_DIR%Base_backend\.env"
    echo GROQ_API_KEY=your_groq_api_key_here >> "%BASE_DIR%Base_backend\.env"
    echo GEMINI_API_KEY=your_gemini_api_key_here >> "%BASE_DIR%Base_backend\.env"
    echo OPENAI_API_KEY=your_openai_api_key_here >> "%BASE_DIR%Base_backend\.env"
    echo ENVIRONMENT=development >> "%BASE_DIR%Base_backend\.env"
    echo DEBUG=True >> "%BASE_DIR%Base_backend\.env"
    echo ✅ Created Base_backend/.env
) else (
    echo ✅ Base_backend/.env already exists
)

REM Create .env file for Dedicated Chatbot
echo [2/8] Setting up Dedicated Chatbot environment...
if not exist "%BASE_DIR%dedicated_chatbot_service\.env" (
    echo # Dedicated Chatbot Environment Variables > "%BASE_DIR%dedicated_chatbot_service\.env"
    echo GROQ_API_KEY=your_groq_api_key_here >> "%BASE_DIR%dedicated_chatbot_service\.env"
    echo OPENAI_API_KEY=your_openai_api_key_here >> "%BASE_DIR%dedicated_chatbot_service\.env"
    echo MODEL_NAME=llama3-8b-8192 >> "%BASE_DIR%dedicated_chatbot_service\.env"
    echo ✅ Created dedicated_chatbot_service/.env
) else (
    echo ✅ dedicated_chatbot_service/.env already exists
)

REM Create .env file for Financial Simulator
echo [3/8] Setting up Financial Simulator environment...
if not exist "%BASE_DIR%Financial_simulator\.env" (
    echo # Financial Simulator Environment Variables > "%BASE_DIR%Financial_simulator\.env"
    echo REDIS_URL=redis://localhost:6379 >> "%BASE_DIR%Financial_simulator\.env"
    echo ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here >> "%BASE_DIR%Financial_simulator\.env"
    echo GROQ_API_KEY=your_groq_api_key_here >> "%BASE_DIR%Financial_simulator\.env"
    echo ✅ Created Financial_simulator/.env
) else (
    echo ✅ Financial_simulator/.env already exists
)

REM Create .env file for Memory Management
echo [4/8] Setting up Memory Management environment...
if not exist "%BASE_DIR%memory_management\.env" (
    echo # Memory Management Environment Variables > "%BASE_DIR%memory_management\.env"
    echo MONGODB_URL=mongodb://localhost:27017 >> "%BASE_DIR%memory_management\.env"
    echo CHROMA_DB_PATH=./chroma_db >> "%BASE_DIR%memory_management\.env"
    echo ✅ Created memory_management/.env
) else (
    echo ✅ memory_management/.env already exists
)

REM Create .env file for Akash Service
echo [5/8] Setting up Akash Service environment...
if not exist "%BASE_DIR%akash\.env" (
    echo # Akash Service Environment Variables > "%BASE_DIR%akash\.env"
    echo GROQ_API_KEY=your_groq_api_key_here >> "%BASE_DIR%akash\.env"
    echo MODEL_NAME=llama3-8b-8192 >> "%BASE_DIR%akash\.env"
    echo ✅ Created akash/.env
) else (
    echo ✅ akash/.env already exists
)

REM Create .env file for Subject Generation
echo [6/8] Setting up Subject Generation environment...
if not exist "%BASE_DIR%subject_generation\.env" (
    echo # Subject Generation Environment Variables > "%BASE_DIR%subject_generation\.env"
    echo WIKIPEDIA_API_URL=https://en.wikipedia.org/api/rest_v1 >> "%BASE_DIR%subject_generation\.env"
    echo KNOWLEDGE_STORE_URL=http://localhost:8003 >> "%BASE_DIR%subject_generation\.env"
    echo ✅ Created subject_generation/.env
) else (
    echo ✅ subject_generation/.env already exists
)

REM Create .env file for Wellness + Forecasting
echo [7/8] Setting up Wellness + Forecasting environment...
if not exist "%BASE_DIR%orchestration\unified_orchestration_system\.env" (
    echo # Wellness + Forecasting Environment Variables > "%BASE_DIR%orchestration\unified_orchestration_system\.env"
    echo GEMINI_API_KEY=your_gemini_api_key_here >> "%BASE_DIR%orchestration\unified_orchestration_system\.env"
    echo GROQ_API_KEY=your_groq_api_key_here >> "%BASE_DIR%orchestration\unified_orchestration_system\.env"
    echo FORECASTING_ENABLED=True >> "%BASE_DIR%orchestration\unified_orchestration_system\.env"
    echo PROPHET_ENABLED=True >> "%BASE_DIR%orchestration\unified_orchestration_system\.env"
    echo ARIMA_ENABLED=True >> "%BASE_DIR%orchestration\unified_orchestration_system\.env"
    echo ✅ Created orchestration/unified_orchestration_system/.env
) else (
    echo ✅ orchestration/unified_orchestration_system/.env already exists
)

REM Create .env file for TTS Service
echo [8/8] Setting up TTS Service environment...
if not exist "%BASE_DIR%tts_service\.env" (
    echo # TTS Service Environment Variables > "%BASE_DIR%tts_service\.env"
    echo TTS_ENGINE=pyttsx3 >> "%BASE_DIR%tts_service\.env"
    echo VOICE_RATE=200 >> "%BASE_DIR%tts_service\.env"
    echo VOICE_VOLUME=0.9 >> "%BASE_DIR%tts_service\.env"
    echo ✅ Created tts_service/.env
) else (
    echo ✅ tts_service/.env already exists
)

echo.
echo ========================================
echo   ENVIRONMENT SETUP COMPLETE
echo ========================================
echo.
echo 🔑 API Key Configuration Required:
echo.
echo Please update the following files with your actual API keys:
echo.
echo   📁 Base_backend/.env
echo      - GROQ_API_KEY (for chatbot functionality)
echo      - GEMINI_API_KEY (for AI features)
echo      - OPENAI_API_KEY (optional, for OpenAI models)
echo.
echo   📁 dedicated_chatbot_service/.env
echo      - GROQ_API_KEY (required for chatbot)
echo.
echo   📁 Financial_simulator/.env
echo      - GROQ_API_KEY (for financial AI)
echo      - ALPHA_VANTAGE_API_KEY (for stock data)
echo.
echo   📁 orchestration/unified_orchestration_system/.env
echo      - GEMINI_API_KEY (for wellness and forecasting AI)
echo.
echo 🔗 How to get API keys:
echo   • Groq API: https://console.groq.com/keys
echo   • Google Gemini: https://makersuite.google.com/app/apikey
echo   • OpenAI: https://platform.openai.com/api-keys
echo   • Alpha Vantage: https://www.alphavantage.co/support/#api-key
echo.
echo 💡 Services will work in limited mode without API keys, but
echo    AI features (chatbot, forecasting AI) will be disabled.
echo.
echo 📋 Next Steps:
echo   1. Add your API keys to the .env files above
echo   2. Run: Backend\install_all_dependencies.bat
echo   3. Run: start_gurukul_with_forecasting.bat
echo.
pause
