# setup_cable.ps1 - one-click VB-CABLE install for livecap
#
#   powershell -ExecutionPolicy Bypass -File setup_cable.ps1
#
# Self-elevates, downloads the driver pack, removes broken duplicate
# cable devices, installs via pnputil (no GUI), offers to reboot.
# After reboot, run livecap.py - status shows "cable OK"; in your call
# app pick "CABLE Output" as the microphone.

param(
    [string]$Url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip",
    [switch]$NoReboot
)

$dst = "$env:USERPROFILE\Downloads\VBCABLE"

# --- self-elevate ---
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "needs admin - relaunching elevated..."
    Start-Process powershell -Verb RunAs -ArgumentList @(
        '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
        $(if ($NoReboot) { '-NoReboot' } else { '' }))
    exit
}

Write-Host "== VB-CABLE setup =="

# --- status check ---
$devices = Get-PnpDevice -PresentOnly |
    Where-Object { $_.FriendlyName -like '*VB-Audio*' }
$good = $devices | Where-Object { $_.Status -eq 'OK' }
$bad  = $devices | Where-Object { $_.Status -ne 'OK' }
foreach ($d in $devices) {
    Write-Host ("  {0}  {1}" -f $d.Status, $d.InstanceId)
}

# --- remove broken duplicates ---
foreach ($d in $bad) {
    Write-Host "removing broken device $($d.InstanceId)"
    pnputil /remove-device $d.InstanceId | Out-Null
}

# --- download + extract ---
New-Item -ItemType Directory -Force $dst | Out-Null
$zip = "$dst\pack.zip"
if (-not (Test-Path "$dst\VBCABLE_Setup_x64.exe")) {
    Write-Host "downloading driver pack..."
    Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing
    Expand-Archive -Force $zip $dst
    Remove-Item $zip
}

# --- verify Authenticode before installing a driver as admin ---
$unsigned = Get-ChildItem $dst -Recurse -Include *.cat,*.sys,*.exe |
    Where-Object {
        (Get-AuthenticodeSignature $_.FullName).Status -ne 'Valid' }
if ($unsigned) {
    Write-Host "ERROR: unsigned or corrupt files in driver pack:"
    $unsigned | ForEach-Object { Write-Host "  $($_.FullName)" }
    exit 1
}

# --- install (skip if a healthy cable already exists) ---
if ($good -and -not $bad) {
    Write-Host "healthy cable device already present - skipping install"
}
else {
    $inf = Get-ChildItem $dst -Filter 'vbMmeCable64_win10.inf' -Recurse |
           Select-Object -First 1 -ExpandProperty FullName
    if ($inf) {
        Write-Host "installing driver: $inf"
        pnputil /add-driver $inf /install
    }
    else {
        Write-Host "win10 inf not found - launching GUI installer"
        Start-Process "$dst\VBCABLE_Setup_x64.exe"
    }
    Write-Host ""
    Write-Host "REBOOT required for CABLE Input/Output to appear"
}

# NOTE: endpoint names can't be renamed programmatically (audiosrv-
# locked; denied even as SYSTEM and via IPropertyStore). The only
# path is mmsys.cpl - so open it and show the user what to click.
if ($good) {
    Start-Process 'control' 'mmsys.cpl,,1'   # Sound -> Recording tab
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Recording tab is open.`n`n" +
        "1. Right-click 'CABLE Output' -> Properties`n" +
        "2. Top text field -> 'livecap Microphone'`n" +
        "3. OK, then Playback tab -> same for 'CABLE Input' -> 'livecap Speaker'",
        'Rename cable endpoints',
        'OK', 'Information') | Out-Null
}

if (-not $NoReboot -and ($bad -or -not $good)) {
    $r = Read-Host "reboot now? [y/N]"
    if ($r -eq 'y') { shutdown /r /t 5 }
}
