# livecap

Real-time, CPU-only Krisp-style app for Windows:
**mic → denoise → virtual mic for calls**, plus **live captions for both
sides** (your voice and speaker output).

- `[them]` — what others say (WASAPI speaker loopback → ASR), with a
  translated line underneath (`--lang`, default `ja`)
- `you:` — what your call actually hears (mic → GTCRN denoise →
  VB-CABLE → tapped for captions)
- Always-on-top overlay, draggable, invisible to screenshots/screen-share

## Setup

Python 3.11; venv is already set up in `.venv/`.

```
.venv\Scripts\python.exe livecap.py           # translates [them] to ja by default
.venv\Scripts\python.exe livecap.py --lang ko # pick another target language
```

For the mic passthrough: install **VB-CABLE** and select "CABLE Output"
as the microphone in your call app. Without it the app warns and still
shows captions for both sources.

## Controls

- Drag the overlay to move it; **Esc** or **Ctrl+C** quits
- `[them]` captions are priority: if the CPU falls behind, `you:` drops first

## Models (in `models/`)

| Piece | Model | Size | Notes |
|---|---|---|---|
| ASR | streaming Zipformer en-2023-06-26 (int8) | ~68 MB | ~4% WER; RTF ≈ 0.5 on 4 cores — only model that sustains real-time here |
| Denoise | GTCRN simple | 0.5 MB | 16 ms frames; RTF ≈ 0.45 |
| Punctuation | online CNN-BiSTM en (int8) | 31 MB | needs lowercase input; restores case |
