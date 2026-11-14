@echo off
rem EquityDesk: double-click to start the market monitor (in the background) and open the app on its Monitor page.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m equitydesk.monitor --background
start "EquityDesk app" /min ".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true
timeout /t 5 /nobreak >nul
start "" "http://localhost:8501/monitor"
