# Workflow (required)

Every change follows: **GitHub issue -> work on a branch -> commit ->
review -> PR -> merge.** Never merge without a review pass — post
self-review notes on the PR before merging; stop before merge if the
user wants to review.

- No `gh` CLI. GitHub REST API via `git credential-manager` token:
  `TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential-manager get | awk '/^password=/{sub(/^password=/,"");print}')`
- Pass JSON bodies via `-d @file` — inline `-d '...'` payloads get mangled.
- Repo: https://github.com/navionis/live-transcribe

# Product: `livecap.py`

Zero-arg Krisp-style app. Run: `.venv\Scripts\python.exe livecap.py`

Two features in one process:

| Leg | Path |
|---|---|
| outbound | mic -> GTCRN denoise -> VB-CABLE virtual mic (what call apps hear) -> tap -> `you:` caption |
| inbound | WASAPI speaker loopback -> streaming zipformer ASR -> `[them]` caption |

## Threading model (5 threads)

```
PortAudio cb (C++ RT)  mic -> mic_q
                                  -> voice_engine thread: GTCRN denoise
                                        |-> Ring buffer -> PortAudio out cb (C++ RT)
                                        |                  -> "CABLE Input" playback
                                        |-> micasr_q (drop-oldest, lowest priority)
soundcard pump thread  loopback -> qs (highest priority)
asr_consumer thread    qs first, micasr_q second -> decode -> gui_q
main thread            tkinter Overlay, after(30ms) drains gui_q
```

- `[them]` captions always win under load: `qs` drains before
  `micasr_q`; on a weak CPU your own dictation drops first
- `voice_engine` is its own thread so outbound voice latency never
  waits behind ASR decode (~35-60ms total mic->cable)
- `Ring` (numpy circular buffer) feeds the cable output callback;
  underrun outputs silence, overflow drops oldest

## Key constants / behavior

- All audio 16kHz mono float32 internally; cable playback resampled to
  the cable's native rate (48k) via `resample_linear`
- ASR = streaming zipformer int8 (`asr_dir()` prefers "zipformer" in
  dir name) + online CNN-BiLSTM punct (needs lowercase input)
- endpoint rules 2.4/1.2s trailing silence, 20s utterance cap
- flush pads 0.75s silence before `input_finished()` — without it
  streaming transducers drop their final tokens
- queues maxsize=512 (~8s at 512-sample chunks); drop-oldest policy
- overlay: frameless always-on-top tkinter, draggable, Esc or Ctrl+C
  to quit; `SetWindowDisplayAffinity(hwnd, 0x11)` hides it from
  screenshots/screen-share (Krisp behavior)
- `--lang CODE` (default `ja`): finalized `[them]` segments translated
  by a worker thread (drain-to-latest) via deep-translator MyMemory,
  falling back to GoogleTranslator; shows on a small accent line under
  `[them]`; `[translation paused - rate limit]` shown on quota errors;
  translations appended to transcript log
- graceful degradation: no VB-CABLE -> captions only; no mic ->
  speaker captions only; loopback error -> shown on `[them]` line

## Hard requirements

- **VB-CABLE** for mic passthrough: program plays into `CABLE Input`,
  call apps must use `CABLE Output` as their mic. Without it the app
  warns and runs captions-only.
- Speaker loopback selected by `m.isloopback and m.id == speaker.id` —
  NOT by name (real mics can share the speaker's name, e.g. "Remote
  Audio" on RDP; `get_microphone(name)` silently picks the wrong one)

## Verified facts (this machine: 4 vCPU Hyper-V, no GPU)

- zipformer int8 RTF ~0.5 — only model sustaining real-time;
  nemotron-0.6b RTF ~2.5 (offline/better-cpu use only)
- GTCRN RTF ~0.45
- `SetWindowDisplayAffinity` works on this build

## Verify

- `.venv\Scripts\python.exe -m py_compile livecap.py`
- live: run, play speech on speaker -> `[them]` tracks it; speak ->
  `you:` shows post-denoise text; Ctrl+C/Esc exits cleanly
