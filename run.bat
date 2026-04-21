@echo off
echo ============================================
echo   SpiroXAI — Lung Disease Diagnostic Demo
echo ============================================
echo.

REM Check if .env exists
if not exist "backend\.env" (
    echo [WARNING] backend\.env not found!
    echo Please copy backend\.env.example to backend\.env and fill in your Supabase credentials.
    echo.
    echo   copy backend\.env.example backend\.env
    echo.
    pause
    exit /b 1
)

echo Starting FastAPI backend on http://localhost:8000 ...
echo Frontend: Open frontend\index.html in your browser
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo.

cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
