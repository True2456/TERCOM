$env:PYTHONPATH="."
$VENV_PYTHON = ".\venv\Scripts\python.exe"

Write-Host "--- Clearing old processes ---" -ForegroundColor Cyan
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "--- Starting TRN System Stack ---" -ForegroundColor Cyan

# 1. Start SITL Simulator
Write-Host "[1/3] Launching SITL Simulator..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='.'; .\venv\Scripts\python.exe simulator\vtol_sim.py" -WindowStyle Normal

# Give SITL a moment to bind ports
Start-Sleep -Seconds 2

# 2. Start TRN Controller
Write-Host "[2/3] Launching Unified TRN Controller..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='.'; .\venv\Scripts\python.exe unified_controller.py" -WindowStyle Normal

# 3. Start Live Map Visualizer
Write-Host "[3/3] Launching Live Map Visualizer..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='.'; .\venv\Scripts\python.exe scripts\live_map_viz.py" -WindowStyle Normal

Write-Host "`nAll components launched. Check the separate PowerShell windows for output." -ForegroundColor Green
Write-Host "If a window closed immediately, there was likely a Python error." -ForegroundColor White
