@echo off
echo ========================================
echo    Gurukul Frontend with Orchestration
echo    Starting React Development Server
echo ========================================
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    npm install
    echo.
)

echo 🚀 Starting React development server...
echo.
echo 🌐 Frontend will be available at: http://localhost:3000
echo.
echo 🎯 Enhanced Features Available:
echo    ✅ AI-Enhanced Lesson Generation
echo    ✅ User Progress Dashboard  
echo    ✅ Smart Intervention System
echo    ✅ RAG-based Content Retrieval
echo.
echo 📋 Make sure Backend services are running:
echo    Run: Backend\start_all_services.bat
echo.

REM Start the development server
npm start

pause
