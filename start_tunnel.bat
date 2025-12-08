@echo off
echo ==========================================
echo   Starting Cloudflare Tunnel
echo ==========================================
echo.
echo Starting tunnel to http://localhost:8000
echo.
echo Your public URL will appear below:
echo.

cloudflared tunnel --url http://localhost:8000

pause
