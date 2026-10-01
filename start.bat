@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   FoodGrade AI
 echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo FoodGrade environment not found.
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

.venv\Scripts\python.exe app.py
pause
