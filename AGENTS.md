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
asr_consumer x2    one thread per stream (spk + mic) -> decode -> gui_q
translator_worker  trans_q -> Google -> MyMemory fallback -> gui_q
hotkey_worker      RegisterHotKey -> WM_HOTKEY -> gui_q
main thread        tkinter Overlay, after(30ms) drains gui_q
```

- Worker loops wrapped in try/except — a dead thread pushes
  `("err", ...)`, which becomes a persistent red badge (not just
  overwritten caption text)
- `Reporter(q).set(comp, lvl, msg)` emits `("state", ...)` only on
  (lvl, msg) transitions — call it every chunk; `ok` on success paths
  clears badges the error path set
- `_poll` reschedules FIRST, tolerates per-event exceptions

## Audio / models

- 16kHz mono float32 internally; cable playback resampled to the
  cable's native rate via the stateful `Resampler` (phase carry +
  ceil n_out so phase stays >= 0 — a floor left it negative at
  step>1 and read future samples); `steer()` closed-loops the rate
  toward a 40ms ring fill (5ms deadband, +-0.1% cap)
- `num_threads = os.cpu_count()` on ASR, denoiser, punct
- ASR provider hardcoded `cpu` — the stock sherpa-onnx pip wheel is
  CPU-only and silently falls back for other provider strings, so
  `make_recognizer` reports what the build actually has
  (`ASR provider: cpu`)
- ASR = streaming zipformer int8 — `asr_dir()` prefers dirs with BOTH
  "zipformer" and "stream" in the name (avoids offline variants);
  punct skipped for nemotron-family dirs
- endpoint rules 2.4/1.2s trailing silence, 20s utterance cap;
  `Pipeline.flush()` pads 0.75s silence before `input_finished()` —
  streaming transducers drop final tokens without it
- queue `get(timeout=0.02)` (20ms); sizes tuned to real latency:
  mic_q=10 (~0.16s — voice path drops, never delays), qs/micasr_q=64
  (~2s/~1s), trans_q=4, gui_q=1024; all puts via `put_drop`
  (drop-oldest) — unbounded gui_q was a real RAM leak
- ring cleared on mute (`ring.clear()`) — buffered voice can't leak;
  `ring.reset(40ms)` refills the jitter buffer on unmute and at open;
  `ring.drop_to(target)` hard-skips a latency burst (>target+60ms)
- single-instance: named mutex `Local\livecap-singleton` first thing
  in `main`; `loading_splash()` shows during model load
- `stream_supervisor` reopens dead mic/cable streams (finished_callback
  or stall -> `mark_dead` -> backoff 1/2/5s, retry logging throttled
  via `_should_log_fail`) — engine picks up new ring/rates from the
  shared `state` dict each chunk, incl. `rs` when a ring appears at
  the SAME rate pair; `pick_mic()` failure is `mark_dead` too so
  hotplug recovers. Dead streams are closed and dropped from `state`
  BEFORE retry — a stale object there blocks `_sd_reinit` (device
  re-enumeration). `_sd_reinit` itself closes whatever's still live
  and marks it dead (~1s passthrough gap) — a replugged device is
  invisible in the frozen PortAudio list without it; `state["cands"]`
  is refreshed post-reinit since indices shift
- `voice_engine` runs unconditionally — a mic/cable that failed at
  startup is reopened by the supervisor and the running engine
  consumes it (its own stall detector does the dead-marking)
- worker loops: per-chunk try/except — one bad chunk badges err and
  continues, thread survives
- stats on stderr every 5s: ring fill + under/over counts, queue
  depths + drops, denoise avg/max/slow-frame ms + fail count,
  per-stream decode RTF (`stat decode spk:rtf=...`)

## Devices

- `cable_candidates()`: PortAudio playback endpoint matching "cable
  input" or "vb-audio point"; prefers WASAPI hostapi, falls back to
  any
- `cable_usable()`: WASAPI-level check — "CABLE Output" must appear
  in `sc.all_microphones()` (RDP sessions hide local endpoints; the
  driver exists but apps can't see it). Fails -> mic leg skipped
- `pick_mic()`: WASAPI host-API default input FIRST (PortAudio's
  global default is MME — tens of ms worse), then sd.default, then
  WASAPI devices, then others — rejects cable/vb-audio
  (self-feedback) + stereo mix/loopback (remote speaker's audio
  would echo back as your voice); None -> `no mic` badge, stream
  never opened. WASAPI mics open with `auto_convert` so Windows
  does the SRC (keeps the resampler out of the mic path). Both
  streams open `latency="low"`, logged via `stream.latency`
- `sd.OutputStream`/`sd.InputStream` open failures degrade to
  captions-only, never crash; `in_stream` assigned only after
  `start()` succeeds

## Overlay

- Frameless always-on-top tkinter; DPI-aware
  (`SetProcessDpiAwareness(1)` system-aware before Tk — Tk 8.6 can't
  track per-monitor anyway)
- Top row: at most ONE badge (err before warn, ~40 chars + `+N`
  for hidden extras) via `plan_badges`; `ok` folds into the summary
  dot (worst level wins), `info` items share one compact chip that
  hides while any err shows; real tk.Button controls pack FIRST so
  badges can't clip them; mic/denoise buttons and `m`/`d` hotkeys
  always exist (`mic_active=True` — a supervisor-reopened leg still
  needs them; a dead mic shows as badges)
- `state` events never touch the caption line — only badges/dot;
  Esc prompt lives in its own `esc_lbl`
- MIC MUTED is a persistent red badge added/removed on toggle
- Global Ctrl+Alt+M / Ctrl+Alt+D via `hotkey_worker`
  (`RegisterHotKey` on its own thread's queue, MOD_NOREPEAT so a
  held key can't toggle, WM_QUIT posted at shutdown) — tk `m`/`d`
  binds only fire with overlay focus, which mid-call it never has;
  a taken chord = warn badge + diag, never fatal. Registered chords
  show as an info chip. NOTE: Ctrl+Alt is AltGr on some layouts —
  the chords can collide with typed characters there (US fine)
- `hidden` / `captured` info chip for
  `SetWindowDisplayAffinity(hwnd, 0x11)` — info on failure too, not
  warn (an amber dot all session would cry wolf). Only blocks
  in-session capture — host-side RDP screenshots still see it
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
  compute/frame; mic->cable total = ~40ms ring target + 16ms frame +
  both stream latencies (see `latency=` diag lines); NOIZEUS 5dB: -9
  to -24dB
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

- `num_threads = os.cpu_count()` — user asked for max-resource use;
  a reviewer-suggested cap was tried and reverted unverified (watch
  `slow=`/`max=` denoise stats if the 16ms deadline ever misses)
- translation on by default (ja) — user's stated requirement
- Esc -> double-press quit, deliberately

## Known deferred work

- shutdown: a worker blocked past its 3s join can outlive the
  transcript log (writes land on a closed file)
- consent defaults (translation sends others' speech out by default)
- typed status model with summary dot (phase 2 of the UX guide)

## Verify

- `.venv\Scripts\python.exe -m py_compile livecap.py`
- `.venv\Scripts\python.exe -m unittest test_livecap` (pure logic —
  no devices)
- `Parser::ParseFile` for .ps1 changes before shipping
- live: play speech -> `[them]` tracks; speak -> `you:` post-denoise
  (when cable usable); Esc x2/Ctrl+C exits clean
