@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ИТП-агент

rem Ollama должен быть запущен: если сервер не отвечает — стартуем его
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo Запускаю Ollama...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    timeout /t 5 /nobreak >nul
)

echo Открываю ИТП-агент в браузере: http://localhost:8501
echo Чтобы остановить — закройте это окно.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false --server.showEmailPrompt false
pause
