$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Write-Host 'Starting dev server on http://localhost:8001'
& $python -X utf8 (Join-Path $PSScriptRoot 'scripts\server.py') --port 8001
