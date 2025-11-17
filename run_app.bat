@echo off
setlocal

REM Change to the directory where this script lives so relative paths work.
cd /d "%~dp0"

REM Delegate the heavy lifting to the Python helper so it can reuse installs.
python run_app.py
if errorlevel 1 goto :error

goto :eof

:error
echo.
echo Failed to prepare or launch the app. Please ensure Python 3.9+ is installed.
exit /b 1

:eof
endlocal
