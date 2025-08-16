# BHIV Core PowerShell Deployment Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🧠 BHIV Core Deployment Script" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "📦 Installing Dependencies..." -ForegroundColor Green
pip install fastapi uvicorn pydantic aiohttp aiofiles
pip install langchain langchain-community langchain-huggingface
pip install sentence-transformers
pip install pytest pytest-asyncio
pip install python-dotenv
pip install requests

Write-Host ""
Write-Host "🔧 Creating Environment File..." -ForegroundColor Green
if (-not (Test-Path ".env")) {
    @"
# BHIV Core Configuration
JWT_SECRET_KEY=bhiv-development-secret-change-in-production
REDIS_URL=redis://localhost:6379
QDRANT_URL=localhost:6333
DATABASE_URL=sqlite:///bhiv_sessions.db
AUDIO_BASE_URL=http://localhost:8010/audio
TTS_SERVICE_URL=http://localhost:8005
DEBUG=true
LOG_LEVEL=INFO
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ Environment file created" -ForegroundColor Green
} else {
    Write-Host "✅ Environment file already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "🧪 Running Tests..." -ForegroundColor Green
python test_bhiv_quick.py

Write-Host ""
Write-Host "🚀 Starting BHIV Core API..." -ForegroundColor Yellow
Write-Host "API will be available at: http://localhost:8010" -ForegroundColor Cyan
Write-Host "Interactive docs at: http://localhost:8010/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

python bhiv_core_api.py
