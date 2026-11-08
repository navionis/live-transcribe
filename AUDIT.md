# livecap — audit guide

Version: main @ `463df40` · Single file: `livecap.py` (~880 lines) +
`build.ps1`, `setup_cable.ps1`, `denoise_file.py`

## 1. What the app does

Two real-time audio legs in one process:

| Leg | Path | Output |
|---|---|---|
| outbound | mic -> GTCRN denoise -> VB-CABLE virtual mic | call apps hear denoised voice |
| outbound tap | same denoised audio -> zipformer ASR | `you:` caption (local) |
| inbound | WASAPI speaker loopback -> zipformer ASR | `[them]` caption (local) |
| optional | finalized `[them]` text -> Google/MyMemory | translation line in overlay |

Everything on screen is drawn by a frameless always-on-top tkinter
overlay; transcripts append to `logs/livecap-*.txt`.

## 2. Data boundary — what leaves the machine

This is the section auditors usually want first:

| Data | Where it goes | Code |
|---|---|---|
| mic audio (raw or denoised) | **nowhere** — in-process numpy buffers, ring buffer -> VB-CABLE driver | `voice_engine` |
| speaker loopback audio | **nowhere** — `qs` queue -> `Pipeline.feed` -> discarded after decode | `loopback_pump` |
| finalized `[them]` text | **external** — deep-translator Google free endpoint, then MyMemory fallback — only when `--lang` is on (default ON) | `translator_worker` |
| partial captions, `you:` captions | never sent | — |
| transcript text (both speakers) | written to disk: `logs/livecap-YYYYMMDD-HHMMSS.txt` | `asr_consumer` |
| translation results | appended to same log file | `translator_worker` |
| window geometry | `logs/livecap-window.json` | `Overlay.close` |

- No audio is ever persisted or transmitted — verified: no file
  writes of audio buffers, no network calls on the audio path
- The only outbound data is finalized `[them]` TEXT, and only via
  deep-translator. No proxy settings honored (see deferred work)
- Logs are plaintext — this is a deliberate user decision, noted as
  a known privacy consideration; retention/clearing not implemented

## 3. Models and execution

| Component | Runtime | Threads | Provider |
|---|---|---|---|
| ASR (streaming zipformer int8, sherpa-onnx) | ONNX Runtime | `os.cpu_count()` | probed `dml -> cuda -> cpu`, printed at startup |
| Denoiser (GTCRN, sherpa-onnx) | ONNX Runtime | `os.cpu_count()` | default |
| Punct (cnn-bilstm, sherpa-onnx) | ONNX Runtime | `os.cpu_count()` | default |

- All inference is local ONNX — no cloud ASR/denoise
- ASR dir chosen by `asr_dir()` — prefers `*zipformer*` + `*stream*`
  names; punct skipped for nemotron-family dirs

## 4. Threading (5 threads)

```
PortAudio cb (RT)    mic -> mic_q(16) -> voice_engine -> Ring -> out cb
                                       \-> micasr_q(64) -------.
soundcard pump     loopback -> qs(64) ----v                     |
asr_consumer       qs first, micasr_q second -> feed -> gui_q(1024)
translator_worker  trans_q(4) -> external APIs -> gui_q
main               tkinter after(30ms) -> _poll drains gui_q
```

- All queues bounded, `put_drop` (drop-oldest) everywhere — the
  unbounded `gui_q` was a real RAM leak, fixed and verified
- Worker loops have per-chunk try/except — one bad chunk logs an
  `err` badge and continues instead of killing the thread
- Speaker wins under load (qs drained first); voice path bounds
  latency at ~0.25 s (mic_q=16 × 16 ms)

## 5. Device selection

- `find_cable()` — PortAudio output matching "cable input"/
  "vb-audio point", WASAPI hostapi preferred
- `cable_usable()` — WASAPI check that "CABLE Output" is enumerable;
  RDP sessions hide local endpoints (driver visible at WDM-KS only),
  in which case mic capture is skipped entirely
- `pick_mic()` — default input first, excludes cable/vb-audio/
  mapper/primary/stereo mix/loopback (feedback + echo-back guards);
  None -> `no mic` badge, no stream opened
- Any device-open failure -> captions-only, never a crash

## 6. Overlay

- Badges = state, one per functional unit (green/warn-red/neutral);
  `("err", ...)` events create persistent red badges
- `SetWindowDisplayAffinity(hwnd, 0x11)` hides from in-session
  capture — badge confirms success, `visible in captures` on failure;
  host-side screenshots still see it
- Esc requires double-press (deliberate — single Esc killed sessions
  mid-call); MIC MUTED persistent badge while muted
- DPI-aware, virtual-screen-clamped geometry, `m`/`d`/buttons inert
  when mic inactive

## 7. Nuitka build

- `build.ps1`: standalone folder build, console forced on;
  `app_root()` resolves `sys.argv[0]` when `__compiled__`; LOGS ->
  `%LOCALAPPDATA%\livecap` when compiled
- Verified: `build\livecap.dist\livecap.exe` runs from foreign cwd,
  models + dml load, degradation works

## 8. Verified measurements (this machine: 6 vCPU Hyper-V)

- ASR per-stream decode RTF **0.08-0.17** live (dml provider) —
  two streams well under real-time
- GTCRN: 16 ms buffer + ~2 ms compute/frame; mic->cable ~45-65 ms
- GTCRN quality (NOIZEUS 5dB): -9 to -24 dB gap noise, 0.999 clean corr
- Captions: partials ~0.3-0.5 s, finals ~1.5-1.7 s after pauses

## 9. Known deferred work (audit-relevant)

- resampler drift: per-chunk `linspace` fills ring ~0.06% fast ->
  ~0.5 s added delay over ~15-25 min call; no fill steering yet
- shutdown: last utterance dropped, no thread joins, log may close
  while a thread writes
- speaker-leg doesn't reconnect if default output changes mid-call
- privacy: translation outbound ON by default; plaintext logs; no
  retention controls; no system-proxy support
- pre-ship: startup preflight for console-off builds, `dml` provider
  confirmation against sherpa-onnx strings
- `flush()` emits nothing (result never read) — cosmetic on exit

## 10. How to verify

```powershell
.\.venv\Scripts\python.exe -m py_compile livecap.py   # compiles
.\.venv\Scripts\python.exe livecap.py                 # run
# stderr shows: ASR model, ASR provider, cable status,
# then every 5s: stat decode {tag}:rtf=... / stat ring=... mic_q=...
```
- play speech on speaker -> `[them]` tracks; speak -> `you:` (when
  cable usable); mute -> MIC MUTED badge, cable goes silent
- Esc x2 quits; transcript lands in `logs/` (or
  `%LOCALAPPDATA%\livecap\logs` for the exe)
- exe: `build\livecap.dist\livecap.exe` from any cwd
