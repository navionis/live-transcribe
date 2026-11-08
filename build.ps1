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
    --company-name=livecap --product-name=livecap --file-version=1.0.0.0 `
    --output-dir=build livecap.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# models ship next to the exe (not embedded - large, swappable).
# robocopy: excludes device-name artifacts (NUL etc.) extracted by tar
robocopy models build\livecap.dist\models /E /XF NUL CON PRN AUX /NFL /NDL /NJH /NJS | Out-Null

Write-Host ""
Write-Host "built: build\livecap.dist\livecap.exe"
Write-Host "test it from another cwd, then flip -windows-console-mode to disable"
