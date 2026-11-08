# Workflow (required)

Every change follows: **GitHub issue -> work on a branch -> commit ->
review -> PR -> merge.** The self-review goes through the real review
API — `POST /pulls/{n}/reviews` with `event: "COMMENT"` and a
non-empty `body` — not an issue comment (those don't count as code
reviews and historically landed on wrong threads). Verify with
`GET /pulls/{n}/reviews` after posting.

- No `gh` CLI. REST API via `git credential-manager` token:
  `TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential-manager get | awk '/^password=/{sub(/^password=/,"");print}')`
- JSON bodies via `-d @file` — inline `-d '...'` gets mangled.
- Commit messages: plain text only — no trailers, no `Co-Authored-By`.
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

- Worker loops wrapped in try/except — a dead thread pushes
  `("err", ...)`, which becomes a persistent red badge (not just
  overwritten caption text)
- `_poll` reschedules FIRST, tolerates per-event exceptions

## Audio / models

- 16kHz mono float32 internally; cable playback resampled to the
  cable's native rate via `resample_linear` (44.1k on this machine)
- `num_threads = os.cpu_count()` on ASR, denoiser, punct
- ASR provider probed per startup: `dml -> cuda -> cpu`, printed as
  `ASR provider: ...`; machine runs `dml`
- ASR = streaming zipformer int8 — `asr_dir()` prefers dirs with BOTH
  "zipformer" and "stream" in the name (avoids offline variants);
  punct skipped for nemotron-family dirs
- endpoint rules 2.4/1.2s trailing silence, 20s utterance cap;
  `Pipeline.flush()` pads 0.75s silence before `input_finished()` —
  streaming transducers drop final tokens without it
- queue `get(timeout=0.02)` (20ms); sizes tuned to real latency:
  mic_q=16 (~0.25s — voice path drops, never delays), qs/micasr_q=64
  (~2s/~1s), trans_q=4, gui_q=1024; all puts via `put_drop`
  (drop-oldest) — unbounded gui_q was a real RAM leak
- ring cleared on mute (`ring.clear()`) — buffered voice can't leak
- worker loops: per-chunk try/except — one bad chunk badges err and
  continues, thread survives
- stats on stderr every 5s: ring fill, queue depths, denoise ms,
  per-stream decode RTF (`stat decode spk:rtf=...`)

## Devices

- `find_cable()`: PortAudio playback endpoint matching "cable input"
  or "vb-audio point"; prefers WASAPI hostapi, falls back to any
- `cable_usable()`: WASAPI-level check — "CABLE Output" must appear
  in `sc.all_microphones()` (RDP sessions hide local endpoints; the
  driver exists but apps can't see it). Fails -> mic leg skipped
- `pick_mic()`: default input FIRST, then others — rejects
  cable/vb-audio (self-feedback) + stereo mix/loopback (remote
  speaker's audio would echo back as your voice); None -> `no mic`
  badge, stream never opened
- `sd.OutputStream`/`sd.InputStream` open failures degrade to
  captions-only, never crash; `in_stream` assigned only after
  `start()` succeeds (buttons can't show for a dead mic)

## Overlay

- Frameless always-on-top tkinter; DPI-aware
  (`SetProcessDpiAwareness(2)` before Tk)
- Top row: status badges (one per functional unit; ✓ -> green,
  warn keywords/"err"/"died" -> red, else gray) + real tk.Button
  controls; mic/denoise buttons and `m`/`d` hotkeys exist only when
  `mic_active`
- MIC MUTED is a persistent red badge added/removed on toggle
- `hidden-from-capture ✓` badge on
  `SetWindowDisplayAffinity(hwnd, 0x11)` success; `visible in
  captures` warn badge on failure. Only blocks in-session capture —
  host-side RDP screenshots still see it
- **Esc requires double-press within 2.5s** (single Esc mid-call was
  a real hazard — intentional, don't "fix" it); Ctrl+C quits
- Partials deduped via `last_p` (push only on text change; entry
  dropped on final); raw zipformer ALL-CAPS sentence-cased
- Geometry persisted to `livecap-window.json`; restored position
  clamped into the virtual screen via `GetSystemMetrics` (all
  monitors incl. negative coords)

## Translation

- `--lang` default `ja`; `lang_code()` normalizes short codes
  (ko->ko-KR, zh->zh-CN, ...) before engine init
- Google and MyMemory each init in their own try — one failure
  doesn't kill the other; Google gets its own code map
  (zh-CN/zh-TW, nb-NO->no)
- finals-only, drain-to-latest queue; quota errors show
  `[translation paused — rate limit]` once; `--trans-email` /
  `LIVECAP_TRANS_EMAIL` raises MyMemory free quota ~10x

## Logs / paths

- `app_root()` returns `sys.argv[0]`'s dir when `__compiled__`
  (Nuitka) else `__file__`'s dir — models resolve against it
- `LOGS`: repo `logs/` in dev, `%LOCALAPPDATA%\livecap\logs` when
  compiled (install dirs may be read-only); `mkdir(parents=True)`
- `.gitignore`: `.venv/`, `models/`, `logs/`, `build/`,
  `test_wavs/`, bytecode, `livecap-*.txt`

## RDP/VM hard limits (verified, this machine)

- Enhanced-session/RDP hides local endpoints — only "Remote Audio"
  visible; Remote Audio and local endpoints are mutually exclusive
  (mic redirect ON hides the cable)
- Endpoint renames impossible: denied as admin (Set-ItemProperty,
  reg.exe), as SYSTEM, and via IPropertyStore::SetValue. Names are
  INF birth names (how Krisp does it). GUI rename via mmsys.cpl only
- So mic->cable->Meet can't fully work in-session; works locally

## Packaging (Nuitka)

`build.ps1` (ASCII-only): pip-installs nuitka, builds
`--standalone` + `--enable-plugin=tk-inter` + include-package for
sherpa_onnx/soundcard/sounddevice/deep_translator, console forced on
(verify before `--windows-console-mode=disable`), then
`robocopy models /E /XF NUL CON PRN AUX` — the nemotron tarball
contains a NUL device-name artifact that kills Copy-Item.
Result: `build\livecap.dist\livecap.exe`, models beside it.

## Verified facts (6 vCPU Hyper-V, no dedicated GPU)

- zipformer int8 RTF ~0.5 — only model sustaining real-time;
  nemotron-0.6b RTF ~2.5 (offline use only). GTCRN RTF ~0.45
- GTCRN: 16ms frame shift + 16ms steady-state buffer + ~2ms
  compute/frame; mic->cable total ~45-65ms; NOIZEUS 5dB: -9 to -24dB
  gap noise, 0.999 clean correlation (see denoise_file.py)
- Captions: partials ~0.3-0.5s, finals ~1.5-1.7s after pauses

## Other files

- `denoise_file.py` — offline GTCRN A/B: file, --batch, or dir;
  prints noise-floor dB + correlation vs `_clean.wav`
- `setup_cable.ps1` — self-elevating: downloads VB-CABLE pack,
  removes broken duplicate devices, pnputil install, opens mmsys.cpl
  with rename instructions. ASCII-only — non-ASCII bytes in strings
  decode as smart quotes under ANSI and break parsing

## Decisions (don't reopen without reason)

- `num_threads = os.cpu_count()` — user asked for max-resource use
- translation on by default (ja) — user's stated requirement
- Esc -> double-press quit, deliberately

## Known deferred work

- resampler drift: chunk-aligned linspace fills the ring ~0.06% fast;
  needs stateful phase carry or soxr
- speaker-leg reconnect when the default output device changes
- shutdown: last utterance lost on exit; no thread joins; log may
  close while threads still write
- consent defaults (translation sends others' speech out by default)
- typed status model with summary dot (phase 2 of the UX guide)

## Verify

- `.venv\Scripts\python.exe -m py_compile livecap.py`
- `Parser::ParseFile` for .ps1 changes before shipping
- live: play speech -> `[them]` tracks; speak -> `you:` post-denoise
  (when cable usable); Esc x2/Ctrl+C exits clean
