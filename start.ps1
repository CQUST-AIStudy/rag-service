[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv not found. Please install uv first." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path (Join-Path $PSScriptRoot ".venv"))) {
  Write-Host ".venv not found. Running uv sync..." -ForegroundColor Yellow
  uv sync
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

Write-Host "Starting RAG service on http://127.0.0.1:8001 ..." -ForegroundColor Cyan
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
exit $LASTEXITCODE
