"""Denoise a 16 kHz mono wav through GTCRN — same model/config livecap uses.

Usage:
    .venv\\Scripts\\python.exe denoise_file.py in.wav [out.wav]
    .venv\\Scripts\\python.exe denoise_file.py test_wavs/noisy16k/*.wav --batch

With --batch, each input becomes <name>_denoised.wav next to the input.
Without --batch and no out arg, writes <stem>_denoised.wav.
"""
import sys
import wave
from pathlib import Path

import numpy as np

from livecap import make_denoiser, SR


def denoise_wav(src: Path, dst: Path) -> float:
    den = make_denoiser()
    with wave.open(str(src), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError(f"{src}: need 16kHz mono wav "
                             f"(got {w.getframerate()}Hz x{w.getnchannels()})")
        a = np.frombuffer(w.readframes(w.getnframes()),
                          np.int16).astype(np.float32) / 32768.0
    shift = den.frame_shift_in_samples
    out = []
    for i in range(0, len(a), shift):
        chunk = a[i:i + shift]
        if len(chunk) < shift:                       # zero-pad tail frame
            chunk = np.concatenate([chunk, np.zeros(shift - len(chunk),
                                                    np.float32)])
        out.extend(den.run(np.ascontiguousarray(chunk), SR).samples)
    out.extend(den.flush().samples)                  # drain buffered tail
    y = np.clip(np.asarray(out, np.float32)[:len(a)], -1.0, 1.0)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return len(y) / SR


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:]]
    batch = "--batch" in argv
    argv = [a for a in argv if a != "--batch"]
    if not argv or (not batch and len(argv) > 2):
        sys.exit(__doc__)
    pairs = [(Path(a), Path(a).with_name(Path(a).stem + "_denoised.wav"))
             for a in argv] if batch or len(argv) == 1 \
        else [(Path(argv[0]), Path(argv[1]))]
    for src, dst in pairs:
        secs = denoise_wav(src, dst)
        print(f"{src} -> {dst}  ({secs:.1f}s)")
