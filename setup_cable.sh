#!/usr/bin/env bash
# setup_cable.sh — one-command VB-CABLE install for livecap (Git Bash)
#
#   ./setup_cable.sh
#
# Downloads the driver pack, removes broken duplicate cable devices,
# installs the driver via pnputil, offers reboot. All admin work runs
# in a single elevated call — one UAC prompt total.
# After reboot: run livecap.py (status shows "cable ✓"), then pick
# "CABLE Output" as the mic in your call app.
set -euo pipefail

URL="https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip"
DST_WIN="$USERPROFILE\\Downloads\\VBCABLE"
DST="$USERPROFILE/Downloads/VBCABLE"

echo "== VB-CABLE setup =="

# --- download + extract (no admin needed) ---
mkdir -p "$DST"
if [ ! -f "$DST/vbMmeCable64_win10.inf" ]; then
    echo "downloading driver pack..."
    curl -sSL -A "Mozilla/5.0" -o "$DST/pack.zip" "$URL"
    unzip -o -q "$DST/pack.zip" -d "$DST"
    rm -f "$DST/pack.zip"
fi

# --- status ---
mapfile -t devs < <(powershell -NoProfile -Command \
  "Get-PnpDevice -PresentOnly | Where-Object { \$_.FriendlyName -like '*VB-Audio*' } | ForEach-Object { \"\$(\$_.Status)|\$(\$_.InstanceId)\" }" \
  | tr -d '\r')
good=0; bad_ids=()
for d in "${devs[@]}"; do
    [ -z "$d" ] && continue
    echo "  $d"
    case "$d" in
        OK\|*)     good=$((good+1)) ;;
        Error\|*)  bad_ids+=("${d#*|}") ;;
    esac
done

if [ "$good" -gt 0 ] && [ ${#bad_ids[@]} -eq 0 ]; then
    echo "healthy cable device present — nothing to install"
    exit 0
fi

# --- one elevated call: remove broken devices + install driver ---
REMOVE=""
for id in "${bad_ids[@]}"; do
    REMOVE+="pnputil /remove-device '$id' | Out-Null; Write-Host 'removed $id'; "
done
INF_WIN="$DST_WIN\\vbMmeCable64_win10.inf"
read -r -d '' PS <<EOF || true
$REMOVE
if (Test-Path '$INF_WIN') {
    Write-Host 'pnputil installing $INF_WIN'
    pnputil /add-driver '$INF_WIN' /install
} else {
    Write-Host 'win10 inf missing — launching GUI installer'
    Start-Process '$DST_WIN\\VBCABLE_Setup_x64.exe'
}
EOF

echo "running admin steps (one UAC prompt)..."
B64=$(powershell -NoProfile -Command \
  "[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(\"$PS\"))" | tr -d '\r\n')
powershell -NoProfile -Command \
  "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile','-EncodedCommand','$B64'"

echo ""
echo ">>> reboot required for CABLE Input/Output to appear <<<"
read -p "reboot now? [y/N] " r
if [ "$r" = "y" ]; then
    B64=$(powershell -NoProfile -Command \
      "[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes('shutdown /r /t 5'))" | tr -d '\r\n')
    powershell -NoProfile -Command \
      "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-EncodedCommand','$B64'"
fi
