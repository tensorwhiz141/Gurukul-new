@echo off
echo ========================================
echo 🧠 BHIV Core Deployment Script
echo ========================================

echo.
echo 📦 Installing Dependencies...
pip install fastapi uvicorn pydantic aiohttp aiofiles
pip install langchain langchain-community langchain-huggingface
pip install sentence-transformers
pip install pytest pytest-asyncio
pip install python-dotenv
pip install requests

echo.
echo 🔧 Creating Environment File...
if not exist .env (
    echo # BHIV Core Configuration > .env
    echo JWT_SECRET_KEY=bhiv-development-secret-change-in-production >> .env
    echo REDIS_URL=redis://localhost:6379 >> .env
    echo QDRANT_URL=localhost:6333 >> .env
    echo DATABASE_URL=sqlite:///bhiv_sessions.db >> .env
    echo AUDIO_BASE_URL=http://localhost:8010/audio >> .env
    echo TTS_SERVICE_URL=http://localhost:8005 >> .env
    echo DEBUG=true >> .env
    echo LOG_LEVEL=INFO >> .env
    echo.
    echo ✅ Environment file created
) else (
    echo ✅ Environment file already exists
)

echo.
echo 🧪 Running Tests...
python test_bhiv_quick.py

echo.
echo 🚀 Starting BHIV Core API...
echo API will be available at: http://localhost:8010
echo Interactive docs at: http://localhost:8010/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python bhiv_core_api.py
