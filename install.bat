@echo off
setlocal
cd /d "%~dp0"

rem Compatibility entry point: keep one Windows installation flow and one venv.
call setup.bat
exit /b %errorlevel%
