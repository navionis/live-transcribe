# Workflow (required)

Every change follows: **GitHub issue -> work on a branch -> commit ->
review -> PR -> merge.** Never merge without a review pass — post
self-review notes on the PR before merging; stop before merge if the
user wants to review.

- No `gh` CLI. GitHub REST API via `git credential-manager` token:
  `TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential-manager get | awk '/^password=/{sub(/^password=/,"");print}')`
- Pass JSON bodies via `-d @file` — inline `-d '...'` payloads get mangled.
- Repo: https://github.com/navionis/live-transcribe

# Project

- `livecap.py` — the product: zero-arg Krisp clone (captions + mic
  denoise -> VB-CABLE). Run: `.venv\Scripts\python.exe livecap.py`
- `transcribe.py` — dev/test tool with flags (`--file`, `--model`,
  `--denoiser`, `--save-clean`, `--loopback`, `--both`).

## Verified facts (this machine: 4 vCPU Hyper-V, no GPU)

- zipformer int8 RTF ~0.5 — only model that sustains real-time;
  nemotron-0.6b RTF ~2.5 (load it only for offline/better-cpu work)
- GTCRN RTF ~0.45; DPDFNet heavier
- streaming transducers need ~0.75s trailing silence pad before
  `input_finished()` or final tokens are dropped
- speaker loopback: `get_microphone(name)` breaks on duplicate
  capture/render names — select loopback mic by `m.isloopback and
  m.id == speaker.id`
- `SetWindowDisplayAffinity(hwnd, 0x11)` verified working

## Verify

- `.venv\Scripts\python.exe -m py_compile livecap.py transcribe.py`
- `.venv\Scripts\python.exe transcribe.py --file test_wavs/0.wav`
  (expected: full transcript, no dropped tail words)
