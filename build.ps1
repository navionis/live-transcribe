# build.ps1 - Nuitka standalone build for livecap.
# Run: powershell -ExecutionPolicy Bypass -File .\build.ps1
$ErrorActionPreference = 'Stop'
$PY = ".\.venv\Scripts\python.exe"

# deps (idempotent)
& $PY -m pip install --quiet nuitka ordered-set zstandard

# standalone folder build; console ON until verified working
& $PY -m nuitka --standalone `
    --enable-plugin=tk-inter `
    --include-package=sherpa_onnx --include-package-data=sherpa_onnx `
    --include-package=soundcard --include-package-data=soundcard `
    --include-package=sounddevice `
    --include-package=deep_translator `
    --windows-console-mode=force `
    --assume-yes-for-downloads `
    --company-name=livecap --product-name=livecap `
    --output-dir=build livecap.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# models ship next to the exe (not embedded - large, swappable)
Copy-Item -Recurse -Force models build\livecap.dist\

Write-Host ""
Write-Host "built: build\livecap.dist\livecap.exe"
Write-Host "test it from another cwd, then flip -windows-console-mode to disable"
