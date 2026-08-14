@echo off
echo =========================================
echo   GenMed AI - Starting Services
echo =========================================

REM Check dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Train model if needed
IF NOT EXIST "models\model.pkl" (
    echo Training ML model...
    python models\train_model.py
)

REM Start backend
echo Starting FastAPI backend on port 8000...
start /B uvicorn backend.main:app --reload --port 8000

timeout /t 4 /nobreak > nul

REM Start frontend
echo Starting Streamlit on port 8501...
streamlit run frontend\app.py --server.port 8501

pause
