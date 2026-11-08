"""Denoise 16 kHz mono wavs through GTCRN — same model/config livecap uses.

Usage:
    .venv\\Scripts\\python.exe denoise_file.py in.wav [out.wav]
    .venv\\Scripts\\python.exe denoise_file.py --batch a.wav b.wav ...
    .venv\\Scripts\\python.exe denoise_file.py test_wavs/noisy16k      # whole dir

Every input becomes <name>_denoised.wav (unless an out path is given).
Prints per-file noise-floor change (dB) in low-energy frames, and the
correlation vs <name>_clean.wav when a clean reference exists nearby.
"""
import sys
import wave
from pathlib import Path

import numpy as np

from livecap import make_denoiser, SR

FRAME = 1600                                    # 100 ms


def noise_floor(a: np.ndarray) -> float:
    """RMS of the quietest 25% of frames — speech-gap noise estimate."""
    rms = np.array([np.sqrt((a[i:i + FRAME] ** 2).mean())
                    for i in range(0, len(a) - FRAME, FRAME)])
    if not len(rms):
        return 0.0
    k = max(1, len(rms) // 4)
    return float(np.sqrt((np.sort(rms)[:k] ** 2).mean()))


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError(f"{p}: need 16kHz mono wav "
                             f"(got {w.getframerate()}Hz x{w.getnchannels()})")
        return np.frombuffer(w.readframes(w.getnframes()),
                             np.int16).astype(np.float32) / 32768.0


def denoise_wav(den, src: Path, dst: Path) -> tuple[np.ndarray, np.ndarray]:
    a = read_wav(src)
    shift = den.frame_shift_in_samples
    out = []
    for i in range(0, len(a), shift):
        chunk = a[i:i + shift]
        if len(chunk) < shift:
            chunk = np.concatenate([chunk, np.zeros(shift - len(chunk),
                                                    np.float32)])
        out.extend(den.run(np.ascontiguousarray(chunk), SR).samples)
    out.extend(den.flush().samples)
    y = np.clip(np.asarray(out, np.float32)[:len(a)], -1.0, 1.0)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return a, y


def report(src: Path, before: np.ndarray, after: np.ndarray) -> str:
    n0, n1 = noise_floor(before), noise_floor(after)
    msg = ""
    if n0 > 0 and n1 > 0:
        msg += f"  noise {20 * np.log10(n1 / n0):+.0f} dB"
    # match a clean reference: <name>_<noise>.wav -> <name>_clean.wav
    clean = src.with_name(src.stem.split("_")[0] + "_clean.wav")
    if clean.exists() and "_clean" not in src.stem:
        c = read_wav(clean)
        m = min(len(c), len(after))
        msg += f"  vs clean corr={np.corrcoef(c[:m], after[:m])[0, 1]:.3f}"
    return msg


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--batch"]
    batch = len(argv) != len(sys.argv[1:])
    if not argv or (not batch and len(argv) > 2):
        sys.exit(__doc__)

    den = make_denoiser()
    targets = []
    for a in argv:
        p = Path(a)
        targets += sorted(p.glob("*_sn*.wav") or p.glob("*.wav")
                          ) if p.is_dir() else [p]
    targets = [t for t in targets if "_denoised" not in t.stem]
    if not targets:
        sys.exit("no wav files found")

    if not batch and len(argv) == 1 and not Path(argv[0]).is_dir():
        pairs = [(targets[0], Path(argv[1]))] if len(argv) == 2 \
            else [(targets[0],
                   targets[0].with_name(targets[0].stem + "_denoised.wav"))]
    else:
        pairs = [(t, t.with_name(t.stem + "_denoised.wav")) for t in targets]

    for src, dst in pairs:
        den.reset()                              # fresh state per file
        before, after = denoise_wav(den, src, dst)
        print(f"{src.name} -> {dst.name}{report(src, before, after)}")
