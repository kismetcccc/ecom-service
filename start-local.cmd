@echo off
cd /d "%~dp0"
if not exist .env (
  echo Missing .env. Configure it before starting.
  pause
  exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
  echo Start Docker Desktop and wait for the engine, then try again.
  pause
  exit /b 1
)
docker compose -f compose.qdrant.yml -f compose.local.yml up -d --build --wait --wait-timeout 180
if errorlevel 1 (
  echo Startup failed. Check the output above and Docker Desktop logs.
  pause
  exit /b 1
)
start "" http://127.0.0.1:18000
echo Ready: http://127.0.0.1:18000
pause
