$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$spec = Join-Path $PSScriptRoot "packaging\loquela.spec"

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found. Create .venv and install requirements first."
}

& $python -m PyInstaller --noconfirm --clean $spec
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Build complete: $(Join-Path $PSScriptRoot 'dist\Loquela\Loquela.exe')"