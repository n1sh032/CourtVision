param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (!(Test-Path -LiteralPath $python)) {
    throw "Create .venv and install this repository's dependencies as described in README.md."
}
Push-Location $PSScriptRoot
try {
    & $python -m courtvision.cli serve --port $Port
    if ($LASTEXITCODE) { throw "The app stopped with exit code $LASTEXITCODE. Read the error above." }
} finally {
    Pop-Location
}
