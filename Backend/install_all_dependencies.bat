@echo off
echo ========================================
echo   GURUKUL DEPENDENCY INSTALLER
echo ========================================
echo.
echo Installing all required Python dependencies for Gurukul platform...
echo This may take 5-10 minutes depending on your internet connection.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo 🐍 Python version:
python --version
echo.

echo [1/8] 📊 Installing Core Data Science Dependencies...
pip install --upgrade pip
pip install numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0 matplotlib>=3.4.0 seaborn>=0.11.0

echo.
echo [2/8] 🔮 Installing Advanced Forecasting Dependencies...
pip install prophet>=1.1.4 statsmodels>=0.14.0 scikit-learn>=1.3.0

echo.
echo [3/8] 🤖 Installing AI/ML Dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.20.0 sentence-transformers>=2.2.0

echo.
echo [3.5/8] 🔗 Installing LangChain Dependencies (Financial Simulator)...
pip install langchain>=0.1.0 langchain-groq>=0.1.0 langchain-community>=0.0.20
pip install langgraph>=0.0.30 langchain-openai>=0.0.5 langchain-google-genai>=0.0.5

echo.
echo [3.7/8] 🔍 Installing Vector Search Dependencies (Orchestration)...
pip install faiss-cpu>=1.7.4 chromadb>=0.4.0
pip install pinecone-client>=2.2.0 weaviate-client>=3.15.0

echo.
echo [4/8] 🌐 Installing Web Framework Dependencies...
pip install fastapi>=0.95.0 uvicorn>=0.20.0 pydantic>=1.10.0 requests>=2.28.0
pip install flask>=2.2.0 flask-cors>=4.0.0 gunicorn>=20.1.0

echo.
echo [5/8] 📱 Installing API and Communication Dependencies...
pip install redis>=4.5.0 celery>=5.2.0 websockets>=10.4
pip install groq>=0.4.0 openai>=1.0.0 google-generativeai>=0.3.0

echo.
echo [6/8] 🔍 Installing Computer Vision Dependencies...
pip install opencv-python>=4.7.0 pillow>=9.4.0 easyocr>=1.7.0
pip install pytesseract>=0.3.10

echo.
echo [7/8] 🎵 Installing Audio/TTS Dependencies...
pip install pyttsx3>=2.90 gTTS>=2.3.0 pygame>=2.1.0 soundfile>=0.12.0
pip install librosa>=0.10.0 speechrecognition>=3.10.0

echo.
echo [8/8] 🔧 Installing Utility Dependencies...
pip install python-dotenv>=1.0.0 pyyaml>=6.0 jsonschema>=4.17.0
pip install psutil>=5.9.0 schedule>=1.2.0 python-multipart>=0.0.6

echo.
echo ========================================
echo   DEPENDENCY VERIFICATION
echo ========================================
echo.

echo Verifying critical dependencies...

python -c "import easyocr; print('✅ easyocr installed successfully')" 2>nul || echo "❌ easyocr installation failed"
python -c "import redis; print('✅ redis installed successfully')" 2>nul || echo "❌ redis installation failed"
python -c "import langchain_groq; print('✅ langchain_groq installed successfully')" 2>nul || echo "❌ langchain_groq installation failed"
python -c "import faiss; print('✅ faiss installed successfully')" 2>nul || echo "❌ faiss installation failed"
python -c "import prophet; print('✅ prophet installed successfully')" 2>nul || echo "❌ prophet installation failed"
python -c "import statsmodels; print('✅ statsmodels installed successfully')" 2>nul || echo "❌ statsmodels installation failed"
python -c "import sklearn; print('✅ scikit-learn installed successfully')" 2>nul || echo "❌ scikit-learn installation failed"
python -c "import fastapi; print('✅ fastapi installed successfully')" 2>nul || echo "❌ fastapi installation failed"
python -c "import torch; print('✅ pytorch installed successfully')" 2>nul || echo "❌ pytorch installation failed"
python -c "import cv2; print('✅ opencv installed successfully')" 2>nul || echo "❌ opencv installation failed"
python -c "import chromadb; print('✅ chromadb installed successfully')" 2>nul || echo "❌ chromadb installation failed"

echo.
echo ========================================
echo   INSTALLATION COMPLETE
echo ========================================
echo.
echo 🎉 All dependencies have been installed!
echo.
echo 📋 Next Steps:
echo   1. Run: start_gurukul_with_forecasting.bat
echo   2. All services should now start without dependency errors
echo   3. Check the service windows for any remaining issues
echo.
echo 💡 If you still see errors:
echo   - Restart your command prompt/terminal
echo   - Run this script as administrator
echo   - Check your Python version (requires 3.8+)
echo.
pause
