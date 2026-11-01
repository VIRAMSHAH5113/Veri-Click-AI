@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  py -3.11 -m venv .venv
  if errorlevel 1 goto :error
)
".venv\Scripts\python.exe" -m pip show fastapi >nul 2>nul
if errorlevel 1 (
  echo Installing runtime requirements...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)
echo Starting Veri-Click AI at http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
exit /b %errorlevel%
:error
echo Setup did not finish. Check that Python 3.11 is installed and available with py -3.11.
pause
exit /b 1
