@echo off
cd /d "%~dp0"
rem Stop the app only; keep the existing shared Qdrant running.
docker compose -f compose.qdrant.yml -f compose.local.yml stop app
pause
