# Workflow (required)

Every change follows: **GitHub issue -> work on a branch -> commit ->
review -> PR -> merge.** Post self-review notes on the PR before
merging; verify the comment landed on the PR (issue/PR comment API
shares `/issues/{n}/comments` — wrong number = wrong thread).

- No `gh` CLI. REST API via `git credential-manager` token:
  `TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential-manager get | awk '/^password=/{sub(/^password=/,"");print}')`
- JSON bodies via `-d @file` — inline `-d '...'` gets mangled.
- Commit messages: plain text only — no "Generated with Devin",
  no `Co-Authored-By` trailers.
- Repo: https://github.com/navionis/live-transcribe

# Product: `livecap.py`

Zero-arg Krisp-style app. Run: `.venv\Scripts\python.exe livecap.py`

| Leg | Path |
|---|---|
| outbound | mic -> GTCRN denoise -> VB-CABLE virtual mic -> `you:` caption |
| inbound | WASAPI speaker loopback -> zipformer ASR -> `[them]` caption |

## Threading model

```
PortAudio cb (RT)    mic -> mic_q -> voice_engine: GTCRN
                         |-> Ring -> PortAudio out cb -> CABLE Input
                         |-> micasr_q (drop-oldest, lowest priority)
soundcard pump     loopback -> qs (highest priority)
asr_consumer       qs first, micasr_q second -> decode -> gui_q
translator_worker  trans_q -> Google -> MyMemory fallback -> gui_q
main thread        tkinter Overlay, after(30ms) drains gui_q
```

## Key constants / behavior

- Audio: 16kHz mono float32 internally; cable resampled to native
  rate (44.1k here) via `resample_linear`
- `num_threads = os.cpu_count()` for ASR, denoiser, punct — scales
  to the machine. Queue `get(timeout=0.02)` — 20ms max queue wait
- endpoint rules 2.4/1.2s trailing silence, 20s utterance cap
- flush pads 0.75s silence before `input_finished()` — without it
  streaming transducers drop final tokens
- queues bounded: mic/micasr/qs=512, trans_q=4, gui_q=1024; all
  puts via `put_drop` — unbounded gui_q was a real RAM leak
- `_poll` reschedules FIRST (immortal drain loop), tolerates
  per-event exceptions
- overlay: frameless always-on-top, draggable, Esc/Ctrl+C quit;
  `SetWindowDisplayAffinity(hwnd, 0x11)` hides from in-session
  screenshots/screen-share (NOT from host-side RDP screenshots)
- status shown as colored badges: functional units each, ✓ -> green,
  warn keywords -> red, else neutral gray; real tk.Button controls
- `--lang` default `ja`; finals-only translation, drain-to-latest,
  `[translation paused — rate limit]` on quota errors; `--trans-email`
  / `LIVECAP_TRANS_EMAIL` raises MyMemory quota ~10x
- logs in `logs/` (transcripts + window json); `.gitignore` covers
  `logs/`, `livecap-*.txt`, `.venv/`, `models/`, `__pycache__/`

## Cable detection (two layers — both required)

- `find_cable()`: PortAudio playback endpoint matching "cable input"
  OR "vb-audio point" (endpoint friendly name differs per driver)
- `cable_usable()`: WASAPI check — `sc.all_microphones()` must list
  "CABLE Output". RDP/Hyper-V sessions hide local endpoints at the
  MMDevice layer: PortAudio still sees the driver (WDM-KS) but apps
  can't. If hidden -> no mic capture at all (`mic off` badge).
- `sd.OutputStream` open failure -> degrade to captions-only too.

## RDP/VM hard limits (verified, this machine)

- Enhanced-session/RDP hides local endpoints: only "Remote Audio"
  visible. Remote Audio (mic redirect) and local endpoints are
  MUTUALLY EXCLUSIVE — enabling RDP mic recording hides the cable.
- Endpoint renames are impossible: denied as admin (Set-ItemProperty,
  reg.exe), as SYSTEM, and via IPropertyStore::SetValue. Names are
  INF birth names (that's how Krisp gets its names). GUI rename via
  mmsys.cpl is the only working path.
- GetDevice-by-ID on hidden endpoint -> fails; object doesn't exist.
- Consequence: mic->denoise->cable->Meet chain cannot fully work in
  an RDP session; works on a local machine.

## Verified facts (4 vCPU Hyper-V, no GPU)

- zipformer int8 RTF ~0.5 — only model sustaining real-time;
  nemotron-0.6b RTF ~2.5 (offline use only). GTCRN RTF ~0.45.
- GTCRN: 16ms frame shift, 16ms steady-state buffer, ~2ms compute/
  frame; mic->cable total ~45-65ms
- GTCRN quality (NOIZEUS 5dB): -9 to -24dB gap noise, 0.999 clean
  correlation (see denoise_file.py)
- Captions: partials ~0.3-0.5s, finals ~1.5-1.7s after pauses
  (1.2s endpoint silence dominates)

## Other files

- `denoise_file.py` — offline GTCRN A/B: single file, --batch, or
  dir; prints noise-floor dB + correlation vs `_clean.wav`
- `setup_cable.ps1` — self-elevating: downloads VB-CABLE pack,
  removes broken duplicate devices, pnputil install, opens mmsys.cpl
  with rename instructions. ASCII-only (em-dash bytes decode as
  smart quotes under ANSI -> parse errors; keep it ASCII).

## Verify

- `.venv\Scripts\python.exe -m py_compile livecap.py`
- `Parser::ParseFile` for .ps1 changes before shipping
- live: run, play speech -> `[them]` tracks; speak -> `you:` shows
  post-denoise text (when cable usable); Esc/Ctrl+C exits clean
