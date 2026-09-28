@echo off
rem Double-click to start the CSV Insight Agent web app.
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" run_app.py %*
pause
