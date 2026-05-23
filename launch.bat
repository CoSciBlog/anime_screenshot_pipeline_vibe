@echo off
setlocal
cd /d "%~dp0"

if not exist "env\Scripts\python.exe" (
  echo Environment not found. Run setup.bat first.
  exit /b 1
)

env\Scripts\python.exe app\gradio_ui.py
endlocal
