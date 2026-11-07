# live-transcribe

Real-time, CPU-only speech pipeline for Windows:
**mic → denoise → streaming English ASR → punctuation/case**

Mirrors Krisp's architecture (small chained models, all streaming), built entirely
on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). No GPU required.

## Setup

Python 3.11 required (deepfilternet wheels only ship up to cp311 on Windows —
not currently used anyway). The venv is already set up in `.venv/`.

```
.venv\Scripts\python.exe transcribe.py
```

## Usage

```
python transcribe.py                     # mic -> gtcrn denoise -> asr -> punct
python transcribe.py --denoiser dpdfnet  # better denoise, ~2x cpu
python transcribe.py --denoiser none     # no denoise
python transcribe.py --device 1          # pick input device (see below)
python transcribe.py --file test_wavs/0.wav  # process a wav file instead of mic
python transcribe.py --loopback          # transcribe speaker output (WASAPI loopback)
python transcribe.py --both              # mic + speaker, tagged [mic]/[spk]
python transcribe.py --model <dir>       # pick ASR model under models/
python transcribe.py --save-clean x.wav  # dump denoised 16 kHz audio
python transcribe.py --threads 6         # asr inference threads
```

List input devices:

```
.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

## Models (in `models/`)

| Piece | Model | Size | Notes |
|---|---|---|---|
| ASR (default) | streaming Zipformer en-2023-06-26 (int8) | ~68 MB | ~4% WER on LibriSpeech, all-caps output; RTF ≈ 0.5 on 4 cores — the only model that sustains real-time here |
| ASR (alt) | NVIDIA Nemotron streaming en-0.6b 560ms/160ms (int8) | ~525 MB | near-Parakeet quality; native case+punct — but RTF ≈ 2.5 on 4 cores, too heavy for live dictation on this box |
| Denoise | GTCRN simple | 0.5 MB | 16 ms frames, very light |
| Denoise (alt) | DPDFNet baseline | 8.8 MB | 10 ms frames, stronger suppression |
| Punctuation | online CNN-BiSTM en (int8) | 31 MB | needs lowercase input; restores case |

## Upgrade paths

- **Better ASR**: swap in `sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8`
  (same release tag `asr-models`) — newer, more accurate, heavier.
- **Better denoise**: DeepFilterNet3 (`deepfilternet` needs torch; or use its
  Rust `libdf` via the C API) — closest open match to Krisp's denoiser.
- **Two-channel capture** (mic + WASAPI loopback of system audio) gives
  diarization for free later — same trick Krisp's inbound/outbound models use.
