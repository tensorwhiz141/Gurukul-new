@echo off
echo Starting Agent Mind-Auth-Memory Link...
echo.
echo Make sure you have:
echo 1. Updated your .env file with actual Supabase credentials
echo 2. MongoDB running (if using local MongoDB)
echo 3. Vedant's agent API running
echo.

echo Checking for existing processes on port 8004...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8004') do (
    echo Killing existing process %%a...
    taskkill /PID %%a /F >nul 2>&1
)

echo Starting server on http://192.168.0.107:8004
echo API docs available at: http://192.168.0.107:8004/docs
echo Health check: http://192.168.0.107:8004/health
echo.
python main.py
pause
