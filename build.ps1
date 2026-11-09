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
    --windows-console-mode=disable `
    --assume-yes-for-downloads `
    --company-name=livecap --product-name=livecap --file-version=1.0.0.0 `
    --output-dir=build livecap.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ASR model: kroko zipformer is required at runtime - fetch if absent
$mdir = Get-ChildItem models -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'kroko' }
if (-not $mdir) {
    $name = 'sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06'
    $tar = Join-Path $env:TEMP ($name + '.tar.bz2')
    Write-Host "downloading $name ..."
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$name.tar.bz2" `
        -OutFile $tar
    tar -xf $tar -C models
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Remove-Item $tar -Force
}

# models ship next to the exe (not embedded - large, swappable).
# robocopy: excludes device-name artifacts (NUL etc.) extracted by tar
robocopy models build\livecap.dist\models /E /XF NUL CON PRN AUX /XD "*160ms*" "*nemotron*" "*fast-conformer*" "*parakeet*" "*zipformer-en-202*" /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -gt 7) { exit $LASTEXITCODE }   # robocopy: 0-7 = ok

Write-Host ""
Write-Host "built: build\livecap.dist\livecap.exe"
Write-Host "console is OFF - stderr is invisible; check badges for state"
