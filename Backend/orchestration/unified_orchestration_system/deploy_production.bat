@echo off
echo ========================================
echo 🚀 BHIV Core Production Deployment
echo ========================================

echo.
echo 📦 Installing Production Dependencies...
pip install fastapi uvicorn[standard] pydantic
pip install gunicorn
pip install psycopg2-binary
pip install redis
pip install qdrant-client
pip install langchain langchain-community
pip install sentence-transformers
pip install python-dotenv
pip install prometheus-client

echo.
echo 🔧 Creating Production Environment...
if not exist .env.production (
    echo # BHIV Core Production Configuration > .env.production
    echo JWT_SECRET_KEY=CHANGE_THIS_IN_PRODUCTION >> .env.production
    echo REDIS_URL=redis://redis-server:6379 >> .env.production
    echo QDRANT_URL=qdrant-server:6333 >> .env.production
    echo DATABASE_URL=postgresql://user:password@db-server:5432/bhiv_core >> .env.production
    echo AUDIO_BASE_URL=https://your-domain.com/audio >> .env.production
    echo TTS_SERVICE_URL=https://your-tts-service.com >> .env.production
    echo DEBUG=false >> .env.production
    echo LOG_LEVEL=WARNING >> .env.production
    echo WORKERS=4 >> .env.production
    echo HOST=0.0.0.0 >> .env.production
    echo PORT=8010 >> .env.production
    echo.
    echo ✅ Production environment template created
    echo ⚠️  Please update .env.production with your actual values
) else (
    echo ✅ Production environment file exists
)

echo.
echo 🧪 Running Production Tests...
python test_bhiv_quick.py

echo.
echo 🐳 Docker deployment files...
echo Creating Dockerfile...

echo FROM python:3.9-slim > Dockerfile
echo. >> Dockerfile
echo WORKDIR /app >> Dockerfile
echo. >> Dockerfile
echo COPY requirements.txt . >> Dockerfile
echo RUN pip install -r requirements.txt >> Dockerfile
echo. >> Dockerfile
echo COPY . . >> Dockerfile
echo. >> Dockerfile
echo EXPOSE 8010 >> Dockerfile
echo. >> Dockerfile
echo CMD ["uvicorn", "bhiv_core_api:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "4"] >> Dockerfile

echo ✅ Dockerfile created

echo.
echo Creating docker-compose.yml...

echo version: '3.8' > docker-compose.yml
echo. >> docker-compose.yml
echo services: >> docker-compose.yml
echo   bhiv-core: >> docker-compose.yml
echo     build: . >> docker-compose.yml
echo     ports: >> docker-compose.yml
echo       - "8010:8010" >> docker-compose.yml
echo     environment: >> docker-compose.yml
echo       - DATABASE_URL=postgresql://postgres:password@db:5432/bhiv_core >> docker-compose.yml
echo       - REDIS_URL=redis://redis:6379 >> docker-compose.yml
echo       - QDRANT_URL=qdrant:6333 >> docker-compose.yml
echo     depends_on: >> docker-compose.yml
echo       - db >> docker-compose.yml
echo       - redis >> docker-compose.yml
echo       - qdrant >> docker-compose.yml
echo. >> docker-compose.yml
echo   db: >> docker-compose.yml
echo     image: postgres:13 >> docker-compose.yml
echo     environment: >> docker-compose.yml
echo       - POSTGRES_DB=bhiv_core >> docker-compose.yml
echo       - POSTGRES_USER=postgres >> docker-compose.yml
echo       - POSTGRES_PASSWORD=password >> docker-compose.yml
echo     volumes: >> docker-compose.yml
echo       - postgres_data:/var/lib/postgresql/data >> docker-compose.yml
echo. >> docker-compose.yml
echo   redis: >> docker-compose.yml
echo     image: redis:7-alpine >> docker-compose.yml
echo. >> docker-compose.yml
echo   qdrant: >> docker-compose.yml
echo     image: qdrant/qdrant >> docker-compose.yml
echo     ports: >> docker-compose.yml
echo       - "6333:6333" >> docker-compose.yml
echo. >> docker-compose.yml
echo volumes: >> docker-compose.yml
echo   postgres_data: >> docker-compose.yml

echo ✅ docker-compose.yml created

echo.
echo Creating requirements.txt...

echo fastapi>=0.104.0 > requirements.txt
echo uvicorn[standard]>=0.24.0 >> requirements.txt
echo pydantic>=2.5.0 >> requirements.txt
echo aiohttp>=3.9.0 >> requirements.txt
echo aiofiles>=23.2.0 >> requirements.txt
echo sqlalchemy>=2.0.0 >> requirements.txt
echo psycopg2-binary>=2.9.0 >> requirements.txt
echo qdrant-client>=1.7.0 >> requirements.txt
echo redis>=5.0.0 >> requirements.txt
echo langchain>=0.1.0 >> requirements.txt
echo langchain-community>=0.0.10 >> requirements.txt
echo langchain-huggingface>=0.0.1 >> requirements.txt
echo sentence-transformers>=2.2.0 >> requirements.txt
echo pyjwt>=2.8.0 >> requirements.txt
echo python-dotenv>=1.0.0 >> requirements.txt
echo prometheus-client>=0.19.0 >> requirements.txt

echo ✅ requirements.txt created

echo.
echo ========================================
echo 🎯 Production Deployment Ready!
echo ========================================
echo.
echo 📋 Next Steps:
echo 1. Update .env.production with your actual values
echo 2. Deploy using Docker: docker-compose up -d
echo 3. Or deploy manually: uvicorn bhiv_core_api:app --host 0.0.0.0 --port 8010 --workers 4
echo.
echo 🔗 Deployment URLs:
echo - API: http://your-domain:8010
echo - Docs: http://your-domain:8010/docs
echo - Health: http://your-domain:8010/health
echo.
pause
