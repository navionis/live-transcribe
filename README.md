# livecap

Live captions for both sides of your calls, plus a Krisp-style denoised
microphone — all local, all on CPU, no account or subscription.

- **[them]** — captions of what others say (captures speaker output),
  with an optional translation line underneath
- **[you]** — captions of what your call actually hears, after noise
  suppression
- **Virtual mic** — sends your denoised voice to any call app
  (Meet, Teams, Zoom, Discord) through VB-CABLE

---

## Requirements

- Windows 10 2004 or newer, 64-bit
- A mid-range CPU (the app uses all available cores; ~4+ cores recommended)
- **VB-CABLE** (free driver) — only needed for the mic-passthrough leg;
  captions work without it
- Headphones recommended — the denoiser is *not* echo cancellation, so
  speaker audio leaking into your mic can appear as `[you]` captions

## Install

1. Download/unpack `livecap.dist` (or build it — see *For developers*).
2. Install VB-CABLE: run `setup_cable.ps1` as administrator, or install
   it from vb-audio.com manually, then reboot.
3. In your call app's audio settings, pick **CABLE Output** as the
   microphone.

## Run

```
livecap.exe                  # from build\livecap.dist\
```

or from source:

```
.venv\Scripts\python.exe livecap.py
```

A dark overlay appears near the bottom of the screen and stays on top.
If a second copy is already running, the new one refuses to start.

> **Remote Desktop caveat:** inside an RDP/enhanced session Windows hides
> local audio endpoints, so the mic → cable leg can't run — captions for
> `[them]` still work. Run it on the local session for full function.

## The overlay

| Action | How |
|---|---|
| Move | drag anywhere |
| Resize | drag the `◢` corner grip |
| Opacity | mouse wheel |
| Font size | Ctrl + wheel |
| Mute mic | `mic:` button, `m` key, or **Ctrl+Alt+M** anywhere |
| Toggle denoise | `denoise:` button, `d` key, or **Ctrl+Alt+D** anywhere |
| Copy last caption | `copy` button or `c` key |
| Copy transcript | `copy all` button |
| Quit | **Esc twice** within 2.5 s, or the `×` button |

The `m`/`d`/`c` keys need the overlay focused; the **Ctrl+Alt** chords
work from any app, mid-call.

The overlay hides itself from screenshots and screen-share capture —
a `captured` chip means that didn't work (e.g. older Windows).

### Status badges

- A colored **dot** summarizes health: green ok, amber warning, red error.
- At most one badge shows at a time; `+N` means more — **click the badge**
  to list them. `MIC MUTED` is a persistent red badge while muted.
- Small gray chips show quiet info: hotkey chords (`C-A-m·C-A-d`),
  capture-hiding state, and translation status.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--lang CODE` | `ja` | Translate `[them]` captions to this language (`ko`, `vi`, `zh-CN`, ...). Pass `--lang ""` to turn translation off |
| `--log-days N` | `0` (keep all) | Delete transcripts older than N days |
| `--trans-email EMAIL` | – | Raises the free translation fallback quota ~10× (or set `LIVECAP_TRANS_EMAIL`) |

## Privacy

- **Translation is on by default** and sends `[them]` caption text to
  Google Translate (MyMemory as fallback) over the network. Speech-to-
  text itself never leaves the machine. Use `--lang ""` to keep captions
  fully local.
- **Transcripts are written to disk** (`logs\livecap-*.txt` or
  `%LOCALAPPDATA%\livecap\logs` for the exe). They're kept forever
  unless you set `--log-days`.

## Troubleshooting

- **No `[you]` / no passthrough** — check the badges. `no VB-CABLE` or
  `cable hidden` means the driver isn't visible (see RDP caveat above).
  `mic engine error`/`no mic input` persisting means the mic leg is down;
  the app retries on its own — unplug/replug the mic or restart the app.
- **Callers hear nothing** — the call app must use *CABLE Output* as its
  mic, and your mic must not be muted (red badge).
- **Voice sounds thin** — expected: the denoise path is 16 kHz, so calls
  get speech-bandwidth (~8 kHz) audio.
- **Diagnostics** — `logs\livecap-diag.log` (`%LOCALAPPDATA%\livecap\logs`
  for the exe) has stats every 5 s: `slow=`/`max=` are denoise deadline
  misses, `under=`/`over=`/`drop=` are audio underruns/drops, `rtf=` is
  decode speed (< 0.5 is healthy). Paste them if you report an issue.
- **Crashes on start** — a dialog names the missing model file; keep the
  `models/` folder next to the exe.

## For developers

`build.ps1` produces `build\livecap.dist\` (Nuitka standalone).
Internals, threading model, and contribution rules live in `AGENTS.md`.
Tests: `.venv\Scripts\python.exe -m unittest test_livecap`.

| Piece | Model | Size |
|---|---|---|
| ASR | streaming Zipformer en-2023-06-26 (int8) | ~68 MB |
| Denoise | GTCRN simple | 0.5 MB |
| Punctuation | online CNN-BiLSTM en (int8) | 31 MB |
