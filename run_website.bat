@echo off
cd /d "%~dp0"
echo Starting LOCO-Inspect AI Web Application on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
".\.venv\Scripts\python.exe" "PTRS\Industry\app.py"
pause
