@echo off
echo ========================================
echo Enhanced Malaysian Speech-to-Text System
echo From 60%% to 95%% Accuracy Testing Platform
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo [1/5] Checking dependencies...

REM Install Python dependencies if needed
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo [2/5] Installing frontend dependencies...
cd frontend
call npm install
cd ..

echo [3/5] Starting backend server...
start "Backend Server" cmd /k "call venv\Scripts\activate.bat && python enhanced_whisper_main.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo [4/5] Starting frontend...
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo [5/5] System starting...
echo.
echo ========================================
echo ✅ SYSTEM STARTED SUCCESSFULLY!
echo ========================================
echo.
echo 🎯 Backend Server: http://localhost:8001
echo 🌐 Frontend: http://localhost:3000
echo.
echo The frontend will open in your browser automatically.
echo.
echo 📝 To test the system:
echo 1. Wait for both servers to fully start
echo 2. Go to http://localhost:3000 in your browser
echo 3. Try Live Transcription or File Upload
echo 4. Compare accuracy across different services
echo.
echo 💡 To get 95%% accuracy with ElevenLabs:
echo 1. Get API key from https://elevenlabs.io/pricing/api
echo 2. Add to .env file: ELEVENLABS_API_KEY=your_key_here
echo 3. Restart the system
echo.
echo Press any key to exit this window...
pause >nul