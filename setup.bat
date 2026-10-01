@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   FoodGrade AI - First Time Setup
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating the single FoodGrade virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment.
        echo Make sure Python 3.11+ is installed and available with the py command.
        pause
        exit /b 1
    )
) else (
    echo Existing FoodGrade .venv found. Skipping environment creation.
)

echo.
echo Installing/updating required packages...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package installation failed. See the message above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo Run start.bat whenever you want to start FoodGrade AI.
echo.
pause
