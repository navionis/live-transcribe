"""bench_wer.py — word-error-rate comparison across model dirs.

WER = (substitutions + deletions + insertions) / reference words,
computed on normalized text (lowercase, punctuation stripped) so a
model isn't penalized for lacking punctuation — and shown separately
since punct/case is still a real feature difference.

Usage: .venv\\Scripts\\python.exe bench_wer.py [model_dir ...]
"""

import os
import re
import sys
from pathlib import Path

from bench_asr import make_rec, read_wav, bench, SR

# filename -> reference transcript (normalized form is derived below)
NOIZEUS_SP01 = "the birch canoe slid on the smooth planks"
TESTS = []


def norm(t: str) -> list:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).split()


def wer(ref: list, hyp: list) -> float:
    # Levenshtein on word lists
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d = d, [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1,
                       prev[j - 1] + (r != h))
    return d[-1] / max(len(ref), 1)


def main():
    refs = {}
    tt = Path("test_wavs/trans.txt")
    if tt.is_file():
        for line in tt.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]
    tests = []
    for name in ("0.wav", "1.wav"):
        p = Path("test_wavs") / name
        if p.is_file() and name in refs:
            tests.append((p, refs[name]))
    for p in sorted(Path("test_wavs/noisy16k").glob("sp01_*.wav")):
        tests.append((p, NOIZEUS_SP01))
    if not tests:
        sys.exit("no wavs with references found")

    roots = [Path(a) for a in sys.argv[1:]] or sorted(
        d for d in Path("models").iterdir()
        if d.is_dir() and (d / "tokens.txt").is_file())
    threads = max(1, (os.cpu_count() or 4) // 2)

    audio = {p.name: read_wav(p) for p, _ in tests}
    print(f"{'model':58s} " + " ".join(
        f"{p.name[:14]:>15s}" for p, _ in tests) + "   mean")
    for d in roots:
        try:
            rec = make_rec(d, threads)
        except SystemExit as e:
            print(f"{d.name}: SKIP {e}")
            continue
        scores = []
        for p, ref in tests:
            _, text = bench(rec, audio[p.name])
            scores.append(wer(norm(ref), norm(text)))
        print(f"{d.name:58s} " + " ".join(
            f"{s * 100:14.1f}%" for s in scores)
            + f" {sum(scores) / len(scores) * 100:6.1f}%")


if __name__ == "__main__":
    main()
