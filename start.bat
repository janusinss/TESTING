@echo off
echo Starting Forensic DGP Web UI...
call venv\Scripts\activate.bat
start http://127.0.0.1:8000
uvicorn app:app --reload
pause
