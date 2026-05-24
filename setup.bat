@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON=py -3.10"
) else (
  set "PYTHON=python"
)

if not exist "env\Scripts\python.exe" (
  echo Creating Python environment in env...
  %PYTHON% -m venv env || exit /b 1
)

env\Scripts\python.exe -m pip --version >nul 2>nul
if not %errorlevel%==0 (
  echo Repairing pip in the existing Python environment...
  env\Scripts\python.exe -m ensurepip --upgrade || exit /b 1
)

env\Scripts\python.exe -m pip install --upgrade pip uv || exit /b 1
git submodule update --init --recursive || exit /b 1

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo Installing CUDA 12.8 PyTorch wheels for NVIDIA GPUs...
  env\Scripts\python.exe -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
) else (
  echo NVIDIA GPU not detected. Installing CPU-compatible PyTorch wheels...
  env\Scripts\python.exe -m uv pip install torch torchvision torchaudio || exit /b 1
)

env\Scripts\python.exe -m uv pip install -r requirements.txt || exit /b 1
env\Scripts\python.exe -m uv pip install -e waifuc || exit /b 1
env\Scripts\python.exe -m uv pip install -e . || exit /b 1

echo Setup completed. Start the UI with launch.bat.
endlocal
