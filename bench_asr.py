"""bench_asr.py — streaming RTF benchmark across sherpa-onnx model dirs.

Feeds each test wav to an OnlineRecognizer in 100ms chunks (simulating
live mic cadence), measures wall-clock decode time vs audio duration.

Usage: .venv\\Scripts\\python.exe bench_asr.py [model_dir ...]
       (default: every dir under models/ that has tokens.txt)
"""

import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

SR = 16000
CHUNK = 1600  # 100 ms

# feature_dim by model family — everything NeMo-parakeet/nemotron is
# 128 mel; zipformer + fastconformer-ctc/transducer are 80
def feat_dim(d: Path) -> int:
    n = d.name.lower()
    return 128 if ("nemotron" in n or "parakeet" in n) else 80


def pick(d: Path, prefix: str) -> str:
    for pat in (f"{prefix}*.int8.onnx", f"{prefix}*.onnx"):
        fs = sorted(d.glob(pat))
        if fs:
            return str(fs[0])
    raise SystemExit(f"{d}: no {prefix}*.onnx")


def make_rec(d: Path, threads: int):
    # transducer dirs have encoder+decoder+joiner; nemo-ctc dirs ship a
    # single model onnx
    if list(d.glob("encoder*.onnx")):
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=pick(d, "encoder"),
            decoder=pick(d, "decoder"),
            joiner=pick(d, "joiner"),
            num_threads=threads,
            sample_rate=SR,
            feature_dim=feat_dim(d),
            enable_endpoint_detection=False,
            decoding_method="greedy_search",
            provider="cpu",
        )
    # sherpa ctc packs name the net "model*.onnx"; fall back to any
    # non-decoder/joiner onnx (e.g. ctc-epoch-*.onnx)
    cands = [f for f in sorted(d.glob("model*.onnx"))
             if ".int8" in f.name] \
        or sorted(d.glob("model*.onnx")) \
        or [f for f in sorted(d.glob("*.onnx"))
            if not f.name.startswith(("decoder", "joiner"))]
    if not cands:
        raise SystemExit(f"{d}: no model onnx")
    model = str(cands[0])
    return sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
        tokens=str(d / "tokens.txt"),
        model=model,
        num_threads=threads,
        sample_rate=SR,
        feature_dim=feat_dim(d),
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
        provider="cpu",
    )


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, p
        raw = w.readframes(w.getnframes())
        sw = w.getsampwidth()
    if sw == 2:
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    if sw == 4:
        return np.frombuffer(raw, np.int32).astype(np.float32) / 2**31
    raise SystemExit(f"{p}: unsupported sampwidth {sw}")


def bench(rec, samples: np.ndarray) -> tuple:
    """Stream samples in 100ms chunks; return (decode_s, text)."""
    s = rec.create_stream()
    t0 = time.monotonic()
    for i in range(0, len(samples), CHUNK):
        s.accept_waveform(SR, np.ascontiguousarray(
            samples[i:i + CHUNK], np.float32))
        while rec.is_ready(s):
            rec.decode_stream(s)
    # flush: trailing silence so the last tokens commit
    s.accept_waveform(SR, np.zeros(SR, np.float32))
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    dt = time.monotonic() - t0
    r = rec.get_result(s)
    text = r if isinstance(r, str) else getattr(r, "text", str(r))
    return dt, text.strip()


def main():
    roots = [Path(a) for a in sys.argv[1:]] or sorted(
        d for d in Path("models").iterdir()
        if d.is_dir() and (d / "tokens.txt").is_file())
    wavs = [Path("test_wavs/0.wav"), Path("test_wavs/1.wav"),
            Path("test_wavs/noisy16k/sp01_babble_sn5.wav")]
    wavs = [w for w in wavs if w.is_file()]
    threads = max(1, (os.cpu_count() or 4) // 2)
    print(f"cpu={os.cpu_count()} threads={threads} "
          f"wavs={[w.name for w in wavs]}")

    audio = {w.name: read_wav(w) for w in wavs}
    for d in roots:
        try:
            rec = make_rec(d, threads)
        except SystemExit as e:
            print(f"\n{d.name}: SKIP ({e})")
            continue
        tot_a = tot_d = 0.0
        print(f"\n=== {d.name} (feat={feat_dim(d)}) ===")
        for name, x in audio.items():
            dt, text = bench(rec, x)
            dur = len(x) / SR
            tot_a += dur
            tot_d += dt
            print(f"  {name:24s} {dur:5.1f}s audio  {dt:5.2f}s decode  "
                  f"rtf={dt/dur:5.2f}")
            print(f"    -> {text[:120]}")
        print(f"  TOTAL rtf={tot_d/tot_a:.2f}")


if __name__ == "__main__":
    main()
