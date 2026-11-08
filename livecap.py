#!/usr/bin/env python
"""livecap — Krisp-style live captions + mic denoise passthrough.

  mic (16k) -> GTCRN denoise -> VB-CABLE virtual mic (what call apps hear)
                             -> tap -> [you] caption
  speaker loopback -> streaming ASR -> [them] caption

Needs VB-CABLE installed and selected as the mic in your call app for the
passthrough half; [them] captions work either way — the you: caption and
mic passthrough need the cable too. Esc or close the window to quit.
"""

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

def app_root() -> Path:
    # under Nuitka (esp. --onefile), __file__ points at a temp
    # extraction dir — use the exe's real location instead
    if "__compiled__" in globals():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parent


ROOT = app_root()
PUNCT_DIR = ROOT / "models" / "sherpa-onnx-online-punct-en-2024-08-06"
SR = 16000

stop = threading.Event()


# ---------------------------------------------------------------- models

class Resampler:
    """Stateful linear resampler — carries fractional phase and the last
    input sample across chunks, so total output tracks the true ratio
    (no cumulative drift at 44.1k: 705.6 samples/chunk stays fractional)."""

    def __init__(self, src_sr: int, dst_sr: int):
        self.step = src_sr / dst_sr        # input samples per output
        self.base_step = self.step
        self.phase = 0.0
        self.tail = 0.0                    # last input sample
        self.trim = 0.0                    # integral skew estimate

    def resample(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate(([self.tail], x))
        last = len(buf) - 1
        # ceil keeps phase >= 0 — with floor, step>1 (44.1k -> 16k)
        # leaves phase negative and buf[-2] reads a sample ~16ms in
        # the future at the head of every chunk
        n_out = max(0, int(np.ceil((last - self.phase)
                                   / self.step - 1e-9)))
        if n_out == 0:
            self.tail = float(buf[-1])
            self.phase -= len(x)
            return np.empty(0, np.float32)
        idx = self.phase + self.step * np.arange(n_out)
        i = np.clip(idx.astype(int), 0, last - 1)
        f = idx - i
        y = buf[i] * (1 - f) + buf[i + 1] * f
        self.phase = idx[-1] + self.step - last
        self.tail = float(buf[-1])
        return y.astype(np.float32)

    def steer(self, err: int, sr: int):
        """Integral rate correction toward a target ring fill:
        err>0 = overfull -> larger step (more input consumed per
        output sample) -> fewer samples pushed. trim converges to the
        device clock skew instead of leaving a proportional offset;
        5ms deadband keeps callback jitter from winding it."""
        if abs(err) >= 0.005 * sr:
            self.trim += (err / sr) * 1e-4     # ~16ms cadence
        self.trim = max(-0.001, min(0.001, self.trim))
        self.step = self.base_step * (1 + self.trim)


def asr_dir() -> Path:
    models = [d for d in (ROOT / "models").iterdir()
              if (d / "tokens.txt").is_file()]
    # zipformer is the only bundled model that sustains real-time on this cpu
    models.sort(key=lambda d: (not ("zipformer" in d.name
                                    and "stream" in d.name), d.name))
    if not models:
        raise SystemExit("no ASR model (with tokens.txt) under models/")
    return models[0]


def find_asr_dir():
    """asr_dir() that returns None instead of raising — SystemExit is
    BaseException, not Exception, so it needs naming explicitly.
    Lets preflight report every missing file, not just the first."""
    try:
        return asr_dir()
    except (Exception, SystemExit):
        return None


def make_recognizer(model_dir: Path):
    def pick(prefix):
        for pat in (f"{prefix}*.int8.onnx", f"{prefix}*.onnx"):
            fs = sorted(model_dir.glob(pat))
            if fs:
                return str(fs[0])
        raise SystemExit(f"no {prefix}*.onnx in {model_dir}")

    kw = dict(tokens=str(model_dir / "tokens.txt"),
              encoder=pick("encoder"), decoder=pick("decoder"),
              joiner=pick("joiner"),
              num_threads=os.cpu_count() or 4,
              sample_rate=SR, feature_dim=80,
              enable_endpoint_detection=True,
              rule1_min_trailing_silence=2.4,
              rule2_min_trailing_silence=1.2,
              rule3_min_utterance_length=20.0,
              decoding_method="greedy_search")
    # stock sherpa-onnx pip wheel is CPU-only — every non-"cpu" string
    # silently falls back to CPU in provider.cc, so don't pretend to
    # probe: report what the build actually has.
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(provider="cpu",
                                                       **kw)
    return rec, "cpu"


def make_denoiser():
    cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
    cfg.model.gtcrn.model = str(ROOT / "models" / "gtcrn_simple.onnx")
    cfg.model.num_threads = os.cpu_count() or 1
    return sherpa_onnx.OnlineSpeechDenoiser(cfg)


def make_punctuator():
    cfg = sherpa_onnx.OnlinePunctuationConfig()
    cfg.model_config.cnn_bilstm = str(PUNCT_DIR / "model.int8.onnx")
    cfg.model_config.bpe_vocab = str(PUNCT_DIR / "bpe.vocab")
    cfg.model_config.num_threads = os.cpu_count() or 1
    return sherpa_onnx.OnlinePunctuation(cfg)


def preflight(root: Path, model_dir):
    """Every file sherpa will hard-exit on if missing — checked before
    any sherpa call so the failure is one clear message, not a silent
    process exit in the windowed build. Returns missing paths."""
    missing = []
    models = root / "models"
    if not models.is_dir():
        return [models]
    if not (models / "gtcrn_simple.onnx").is_file():
        missing.append(models / "gtcrn_simple.onnx")
    if model_dir is None:
        missing.append("ASR model dir (with tokens.txt) under models/")
    else:
        if not (model_dir / "tokens.txt").is_file():
            missing.append(model_dir / "tokens.txt")
        for pat in ("encoder*.onnx", "decoder*.onnx", "joiner*.onnx"):
            if not list(model_dir.glob(pat)):
                missing.append(model_dir / pat)
        if "nemotron" not in model_dir.name.lower():
            punct = models / PUNCT_DIR.name
            for f in ("model.int8.onnx", "bpe.vocab"):
                if not (punct / f).is_file():
                    missing.append(punct / f)
    return missing


def result_text(res) -> str:
    return res if isinstance(res, str) else getattr(res, "text", str(res))


_diag_log = [None]
_diag_path = [None]
_diag_size = [0]
_diag_q = queue.Queue(maxsize=512)


def _diag_writer():
    """Drains diag lines to stderr + the log file — writers never block
    on a stalled disk. Rotates by size mid-run, not just at startup."""
    while True:
        try:
            line = _diag_q.get()
        except Exception:
            continue
        if isinstance(line, threading.Event):   # diag_flush sentinel
            line.set()
            continue
        try:                             # stderr can be a non-None
            if sys.stderr is not None:   # object that raises on write
                sys.stderr.write(line + "\n")   # (windowed build) —
        except Exception:                # must not kill file logging
            pass
        try:
            f = _diag_log[0]
            if f is None:
                continue
            f.write(line + "\n")
            _diag_size[0] += len(line) + 1
            if _diag_size[0] > 2 * 1024 * 1024:
                _diag_size[0] = 0
                _diag_log[0] = None
                try:
                    f.close()
                    _diag_path[0].replace(
                        _diag_path[0].with_name("livecap-diag.1.log"))
                    _diag_log[0] = open(_diag_path[0], "a",
                                        encoding="utf-8", buffering=1)
                except Exception:
                    pass
        except Exception:
            pass


threading.Thread(target=_diag_writer, daemon=True,
                 name="diag-writer").start()


def diag(msg):
    """Diagnostics channel — non-blocking: stderr (when it exists;
    windowed builds can have sys.stderr=None) + rotating log."""
    put_drop(_diag_q, f"[{time.strftime('%H:%M:%S')}] {msg}")


def diag_flush(timeout: float = 1.0):
    """Block until every queued diag line is written — shutdown and
    fatal paths call this so the last lines can't race teardown."""
    ev = threading.Event()
    try:
        _diag_q.put_nowait(ev)
    except queue.Full:
        return
    ev.wait(timeout)


def diag_exc(prefix, t, v, tb):
    """One diag line with the full traceback — 'thread X died:
    KeyError' alone can't be diagnosed."""
    diag(f"{prefix}: " + "".join(traceback.format_exception(t, v, tb)))


def put_drop(q: queue.Queue, item):
    """Non-blocking put; on full, drop the oldest chunk to bound
    latency. Drops are counted on the queue (q.drops) and reported to
    the caller — silent loss is how latency bugs hide."""
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        pass
    dropped = False
    try:
        q.get_nowait()
        dropped = True                       # evicted the oldest
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        dropped = True                       # item itself dropped
    if dropped:
        q.drops = getattr(q, "drops", 0) + 1
    return dropped


class Reporter:
    """Emits ('state', comp, lvl, msg) only on transitions — callers can
    report on every chunk/decode without flooding gui_q, and recovery
    ('ok') clears badges the error path set."""

    def __init__(self, q):
        self.q, self.last = q, {}

    def set(self, comp, lvl, msg=""):
        if self.last.get(comp) != (lvl, msg):
            self.last[comp] = (lvl, msg)
            put_drop(self.q, ("state", comp, lvl, msg))


def badge_shown(lvl: str) -> bool:
    """Only warn/err get badges — ok folds into the dot, info into the
    single compact chip, so the status row stays inside the window."""
    return lvl in ("warn", "err")


_BADGE_PRIORITY = ("mute", "mic", "cable", "speaker")


def plan_badges(lvl_map, txt_map):
    """State maps -> (shown_comps, hidden_count, chip_text). The badge
    row has ~200px after the buttons, so: at most one badge (err
    before warn, _BADGE_PRIORITY order within a level — mute must not
    hide behind another err) plus a '+N' count for the rest, and the
    info chip hides while any err is up."""
    errs = [c for c, l in lvl_map.items() if l == "err"]
    warns = [c for c, l in lvl_map.items() if l == "warn"]
    rank = lambda c: (_BADGE_PRIORITY.index(c)
                      if c in _BADGE_PRIORITY else len(_BADGE_PRIORITY))
    errs.sort(key=rank)                  # stable — ties keep
    warns.sort(key=rank)                 # insertion order
    shown = (errs or warns)[:1]
    extra = len(errs) + len(warns) - len(shown)
    chip = "" if errs else " · ".join(
        t for c, t in txt_map.items() if lvl_map.get(c) == "info" and t)
    return shown, extra, chip


def lang_code(code: str) -> str:
    """Normalize 'ko' -> 'ko-KR' style codes for the translator."""
    code = code.strip().lower()
    special = {
        "zh": "zh-CN", "zh-tw": "zh-TW", "zh-hk": "zh-HK",
        "en": "en-US", "pt": "pt-PT", "no": "nb-NO",
        # lang != region for these common targets
        "ko": "ko-KR", "ja": "ja-JP", "vi": "vi-VN", "fr": "fr-FR",
        "de": "de-DE", "es": "es-ES", "it": "it-IT", "nl": "nl-NL",
        "sv": "sv-SE", "da": "da-DK", "fi": "fi-FI", "pl": "pl-PL",
        "ru": "ru-RU", "uk": "uk-UA", "cs": "cs-CZ", "sk": "sk-SK",
        "hu": "hu-HU", "ro": "ro-RO", "bg": "bg-BG", "hr": "hr-HR",
        "sl": "sl-SI", "lt": "lt-LT", "lv": "lv-LV", "et": "et-EE",
        "el": "el-GR", "tr": "tr-TR", "ar": "ar-SA", "he": "he-IL",
        "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
        "ur": "ur-PK", "pa": "pa-IN", "gu": "gu-IN", "kn": "kn-IN",
        "ml": "ml-IN", "mr": "mr-IN", "si": "si-LK", "th": "th-TH",
        "my": "my-MM", "km": "km-KH", "lo": "lo-LA", "ms": "ms-MY",
        "id": "id-ID", "tl": "tl-PH", "mn": "mn-MN", "kk": "kk-KZ",
        "ka": "ka-GE", "hy": "hy-AM", "az": "az-AZ", "uz": "uz-UZ",
        "sq": "sq-AL", "mk": "mk-MK", "sr": "sr-Latn-RS", "fa": "fa-IR",
        "sw": "sw-KE", "af": "af-ZA", "am": "am-ET",
    }
    if code in special:
        return special[code]
    if "-" in code:                      # ll-rr -> ll-RR (zh-cn -> zh-CN)
        lang, _, region = code.partition("-")
        return f"{lang}-{region.upper()}"
    return f"{code}-{code.upper()}"


def translator_worker(tq, gui_q, log, log_lock, lang, email):
    """Translate finalized [them] segments off the caption critical path."""
    try:
        from deep_translator import GoogleTranslator, MyMemoryTranslator
    except Exception as e:
        put_drop(gui_q, ("trans", f"[translation unavailable: {e}]"))
        return
    gmap = {"zh-CN": "zh-CN", "zh-TW": "zh-TW", "zh-HK": "zh-TW",
            "nb-NO": "no", "he-IL": "iw"}   # he/zh-HK rejected by Google
    gcode = gmap.get(lang, lang.split("-")[0])
    try:
        tr = GoogleTranslator(source="en", target=gcode)
    except Exception as e:
        tr = None
        put_drop(gui_q, ("trans", f"[google init failed: {e}]"))
    try:
        fallback = MyMemoryTranslator(source="en-US", target=lang,
                                      email=email)
    except Exception as e:
        fallback = None
        put_drop(gui_q, ("trans", f"[mymemory init failed: {e}]"))
    if tr is None and fallback is None:
        put_drop(gui_q, ("state", "trans", "warn", "translation off"))
        return
    warned = False
    while not stop.is_set():
        try:
            sid, text = tq.get(timeout=0.1)
        except queue.Empty:
            continue
        while True:                        # keep only the newest segment
            try:
                item = tq.get_nowait()
                with log_lock:             # name the dropped seg id
                    log.write(f"    -> (skipped seg {sid})\n")
                sid, text = item
            except queue.Empty:
                break
        out, reason = None, None
        try:
            if tr is not None:
                out = tr.translate(text)
        except Exception as e:
            reason = type(e).__name__
        if out is None and fallback is not None:
            try:
                out = fallback.translate(text)
            except Exception as e:
                reason = reason or type(e).__name__
        if out:
            warned = False
            put_drop(gui_q, ("trans", out))
            with log_lock:
                log.write(f"    -> {out}  (seg {sid})\n")
        elif not warned:                   # visible once, not per call
            warned = True
            put_drop(gui_q, ("trans",
                             f"[translation paused: {reason or 'unknown'}]"))


# ------------------------------------------------------------------- asr

class Pipeline:
    def __init__(self, recognizer, punct):
        self.rec = recognizer
        self.punct = punct
        self.streams = {}
        self.partials = {}
        self.lock = threading.Lock()   # serialize decode across streams
        self.plock = threading.Lock()  # shared punctuator isn't mt-safe

    def _punct(self, text: str) -> str:
        if not self.punct or not text:
            return text
        with self.plock:
            return self.punct.add_punctuation_with_case(text.lower())

    def feed(self, samples: np.ndarray, tag: str):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if not len(samples):
            return
        # one recognizer, two consumers — every call on it (create,
        # decode, endpoint, result, reset) runs under the lock
        text = None
        with self.lock:
            stream = self.streams.get(tag)
            if stream is None:
                stream = self.streams[tag] = self.rec.create_stream()
            stream.accept_waveform(SR, samples)
            while self.rec.is_ready(stream):
                self.rec.decode_stream(stream)
            if self.rec.is_endpoint(stream):
                text = result_text(
                    self.rec.get_result(stream)).strip()
                self.rec.reset(stream)
                self.partials.pop(tag, None)
            else:
                self.partials[tag] = result_text(
                    self.rec.get_result(stream)).strip()
        # punctuation is slow and has its own lock — holding the decode
        # lock through it would stall the other stream
        if text:
            return self._punct(text)

    def flush(self, tag):
        """Force-finalize one stream (per consumer); returns pending
        text or None. Padding silence is required — streaming
        transducers drop final tokens without it."""
        with self.lock:                  # both consumers flush at exit
            stream = self.streams.get(tag)
            if stream is None:
                return None
            pad = np.zeros(int(0.75 * SR), dtype=np.float32)
            stream.accept_waveform(SR, pad)
            stream.input_finished()
            while self.rec.is_ready(stream):
                self.rec.decode_stream(stream)
            text = result_text(self.rec.get_result(stream)).strip()
            self.streams.pop(tag, None)
        return self._punct(text) or None


# ------------------------------------------------------ mic -> denoise ->
#                                                cable + [you] caption tap

class Ring:
    """Float32 ring buffer feeding the cable output callback."""

    def __init__(self, n: int):
        self.b = np.zeros(n, np.float32)
        self.n = n
        self.r = self.w = 0
        self.underruns = self.overruns = 0
        self._armed = False              # underruns count only once
        self.lock = threading.Lock()     # real audio has flowed

    def push(self, x: np.ndarray):
        n = len(x)
        if n > self.n:                    # keep only the newest samples
            x = x[n - self.n:]
            n = self.n
        with self.lock:
            if n > self.n - (self.w - self.r):   # overflow: drop oldest
                self.overruns += 1
                self.r += n - (self.n - (self.w - self.r))
            i = self.w % self.n
            k = min(n, self.n - i)
            self.b[i:i + k] = x[:k]
            self.b[:n - k] = x[k:]
            self.w += n
            self._armed = True

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self.lock:
            have = min(n, self.w - self.r)
            if have < n and self._armed:
                self.underruns += 1
            i = self.r % self.n
            k = min(have, self.n - i)
            out[:k] = self.b[i:i + k]
            out[k:have] = self.b[:have - k]
            self.r += have
        return out

    def fill(self) -> int:
        return self.w - self.r

    def clear(self):
        """Drop buffered audio — called on mute so nothing leaks out.
        Disarms underrun counting; muted pulls aren't failures."""
        with self.lock:
            self.r = self.w
            self._armed = False

    def reset(self, prefill: int = 0):
        """Clear, then refill with silence — used at open and after
        unmute so the output callback always has jitter headroom."""
        with self.lock:
            self.r = self.w
        if prefill:
            self.push(np.zeros(prefill, np.float32))


def cable_candidates():
    """Ordered (index, sr, wasapi?) candidates for the VB-CABLE
    playback endpoint — 'CABLE Input' or 'Speakers (VB-Audio Point)'.
    WASAPI first (auto-convert handles mono-on-stereo shared mode),
    then other host APIs."""
    apis = sd.query_hostapis()
    devs = sd.query_devices()
    def ok(d):
        n = d["name"].lower()
        return ("cable input" in n or "vb-audio point" in n) \
            and d["max_output_channels"] > 0
    out, order = [], ("Windows WASAPI", "Windows MME",
                      "Windows DirectSound")
    for api in order:
        for i, d in enumerate(devs):
            if ok(d) and apis[d["hostapi"]]["name"] == api:
                out.append((i, int(d["default_samplerate"]),
                            api == "Windows WASAPI"))
    for i, d in enumerate(devs):
        tup = (i, int(d["default_samplerate"]), False)
        if ok(d) and tup not in out:
            out.append(tup)
    return out


def wasapi_default_input():
    """The WASAPI host API's own default input. PortAudio's global
    default on Windows is MME, and MME 'low' latency is still tens of
    ms worse than WASAPI — so this one goes first, not sd.default."""
    for a in sd.query_hostapis():
        if a["name"] == "Windows WASAPI":
            d = a["default_input_device"]
            return d if d >= 0 else None
    return None


def pick_mic():
    """Mic index: WASAPI default first, then PortAudio's default, then
    WASAPI devices, then others — skipping cable/vb-audio (feedback)
    and stereo mix/loopback (remote speaker's audio would go out as
    your voice)."""
    bad = ("cable", "vb-audio", "mapper", "primary", "stereo mix",
           "loopback")
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    ok = lambda i: devs[i]["max_input_channels"] > 0 and not any(
        b in devs[i]["name"].lower() for b in bad)
    wasapi = lambda i: apis[devs[i]["hostapi"]]["name"] \
        == "Windows WASAPI"          # lower latency than the MME path
    first = [i for i in (wasapi_default_input(), sd.default.device[0])
             if isinstance(i, int) and i >= 0]
    cands = first + [i for i in range(len(devs)) if wasapi(i)] \
        + list(range(len(devs)))
    return next((i for i in cands if ok(i)), None)


def cable_usable() -> bool:
    """WASAPI-level check: can apps actually pick 'CABLE Output' as a
    mic? RDP sessions hide local endpoints — PortAudio still sees the
    cable there, but it's useless, so the mic leg gets skipped."""
    try:
        import soundcard as sc
        return any("cable output" in m.name.lower()
                   for m in sc.all_microphones())
    except Exception:
        return False


def open_cable(cands, usable, open_fn):
    """Try cable candidates in order. open_fn(idx, sr, wasapi, ring)
    returns a stream or raises. Never raises — returns
    (None, None, SR) on no candidates / hidden / all-fail so the
    captions-only path still gets a valid cable_sr."""
    if not usable:
        return None, None, SR
    for idx, sr, wasapi in cands:
        try:
            ring = Ring(sr // 2)            # 0.5s buffer
            ring.reset(int(0.04 * sr))      # 40ms jitter prefill
            stream = open_fn(idx, sr, wasapi, ring)
            diag(f"cable open: dev{idx} "
                 f"{'wasapi+autoconv' if wasapi else 'plain'}")
            return ring, stream, sr
        except Exception as e:
            diag(f"cable candidate {idx} failed: {e}")
    return None, None, SR


def open_mic(mic_idx, den, mic_q, gui_q):
    """Open the mic stream: 16kHz first (WASAPI gets auto_convert so
    Windows does the SRC and the resampler stays out of the mic
    path), then the device's native rate — with auto_convert too,
    since a WASAPI fallback usually failed on channels, not rate.
    finished_callback turns a dead device into a badge instead of
    silence. Returns (stream, mic_sr); raises if both fail."""
    kw = {}
    try:
        if sd.query_hostapis()[sd.query_devices(
                mic_idx)["hostapi"]]["name"] == "Windows WASAPI":
            kw["extra_settings"] = sd.WasapiSettings(auto_convert=True)
    except Exception:
        pass
    cb = lambda d, f, t, s: put_drop(mic_q, d[:, 0].copy())
    stopped = lambda: put_drop(
        gui_q, ("state", "mic", "err", "mic stream stopped"))
    try:
        return sd.InputStream(
            device=mic_idx, samplerate=SR, channels=1,
            dtype="float32", latency="low",
            blocksize=den.frame_shift_in_samples, callback=cb,
            finished_callback=stopped, **kw), SR
    except Exception as e:
        diag(f"mic 16k open failed ({e}); trying native rate")
    mic_sr = int(sd.query_devices(mic_idx)["default_samplerate"])
    return sd.InputStream(
        device=mic_idx, samplerate=mic_sr, channels=1,
        dtype="float32", latency="low",
        blocksize=int(den.frame_shift_in_samples * mic_sr / SR),
        callback=cb, finished_callback=stopped, **kw), mic_sr


def _should_log_fail(n: int) -> bool:
    """Bound denoise-failure diag lines: first few, then one per
    200 — a persistent failure can't flood the 2MB log."""
    return n <= 3 or n % 200 == 0


def voice_stat(ring, mic_q, micasr_q, avg_ms, max_ms, slow, fails):
    """One stderr stats line — surfaces what an average hides:
    underruns, overruns, queue evictions, deadline misses, denoise
    failure count."""
    return (f"stat ring={ring.fill() if ring else 0} "
            f"under={ring.underruns if ring else 0} "
            f"over={ring.overruns if ring else 0} "
            f"mic_q={mic_q.qsize()} drop={getattr(mic_q, 'drops', 0)} "
            f"micasr={micasr_q.qsize()} "
            f"denoise={avg_ms:.1f}ms max={max_ms:.1f}ms "
            f"slow={slow} fails={fails}")


def voice_engine(mic_q, den, ring, micasr_q, mute, den_off, gui_q,
                 mic_sr=SR, cable_sr=SR, stall_after=1.5):
    """mic -> denoise -> ring buffer for cable + tap into ASR queue."""
    rep = Reporter(gui_q)                    # dedupes; ok clears badges
    needs_reset = muted = False
    mic_rs = Resampler(mic_sr, SR) if mic_sr != SR else None
    pend = np.empty(0, np.float32)           # native-rate accumulator
    rs = Resampler(SR, cable_sr) if ring is not None else None
    target = int(0.04 * cable_sr)            # ~40ms target ring fill
    shift = den.frame_shift_in_samples
    stat_t, stat_dn, stat_ms = time.time(), 0, 0.0
    stat_max = stat_slow = den_fails = 0
    last_chunk = time.time()
    while not stop.is_set():
        try:
            x = mic_q.get(timeout=0.02)
            last_chunk = time.time()
        except queue.Empty:
            # chunks keep arriving even while muted — a stall means
            # the device is gone, not silent
            if time.time() - last_chunk > stall_after:
                rep.set("mic", "err", "no mic input")
            continue
        if mic_rs is not None:               # native-rate input -> 16k
            pend = np.concatenate((pend, mic_rs.resample(x)))
            n = len(pend) - len(pend) % shift
            if not n:                        # hosts can send blocks
                continue                     # bigger than one frame —
            x, pend = pend[:n], pend[n:]     # drain all full frames
        try:
            if mute.is_set():                # hard mute: nothing leaves
                needs_reset = muted = True
                if ring is not None:
                    ring.clear()             # drain buffered voice too
                continue
            if muted:                        # unmute: refill the jitter
                muted = False                # buffer or the first
                if ring is not None:         # bursts underrun
                    ring.reset(target)
            if den_off.is_set():             # passthrough without denoise
                y = np.ascontiguousarray(x, np.float32)
                needs_reset = True
                rep.set("denoise", "ok")
            else:
                t0 = time.monotonic()
                try:
                    if needs_reset:          # resync denoiser frame
                        try:                 # state; a failed reset
                            den.reset()      # must not stick raw mode
                        except Exception as e:
                            diag(f"denoise reset failed: {e}")
                        needs_reset = False
                    out = den.run(np.ascontiguousarray(x, np.float32), SR)
                    ms = (time.monotonic() - t0) * 1000
                    stat_ms += ms
                    stat_dn += len(x) / shift    # count 16ms frames
                    per = ms / (len(x) / shift)  # ~ms per 16ms frame
                    stat_max = max(stat_max, per)
                    if per > 12:               # misses the 16ms deadline
                        stat_slow += 1
                    y = np.asarray(out.samples, dtype=np.float32)
                    rep.set("denoise", "ok")
                except Exception as e:       # fail open: raw chunk out
                    den_fails += 1
                    y = np.ascontiguousarray(x, np.float32)
                    needs_reset = True
                    rep.set("denoise", "err", "bypassed: raw audio")
                    # a persistent failure must not flood the log —
                    # first few, then one line per 200
                    if _should_log_fail(den_fails):
                        diag(f"denoise fail x{den_fails}: {e}")
            if not len(y):
                continue
            if ring is not None:
                fill = ring.fill()
                rs.steer(fill - target, cable_sr)  # ~40ms jitter buffer
                ring.push(rs.resample(y))
            put_drop(micasr_q, y)
            rep.set("mic", "ok")
            if time.time() - stat_t > 5:
                if rs is not None:
                    # saturated AND still off-target = real skew;
                    # trim alone at the cap is just windup
                    if abs(rs.trim) > 0.0009 and abs(
                            ring.fill() - target) > 0.01 * cable_sr:
                        rep.set("cable", "err", "clock skew saturating")
                    else:
                        rep.set("cable", "ok")
                diag(voice_stat(ring, mic_q, micasr_q,
                                stat_ms / max(stat_dn, 1), stat_max,
                                stat_slow, den_fails))
                stat_t, stat_dn, stat_ms = time.time(), 0, 0.0
                stat_max = stat_slow = 0
        except Exception as e:
            rep.set("mic", "err", f"mic engine error: {e}")


# -------------------------------------------------------- speaker -> [them]

def loopback_pump(qs, gui_q):
    """Supervised speaker-loopback pump: re-resolves the default output
    every ~2s, reopens on device change or recorder error (backoff),
    and reports state — [them] can't die silently on a headset switch."""
    import ctypes
    com_init = False
    try:
        import warnings
        import soundcard as sc
        ctypes.windll.ole32.CoInitialize(None)
        com_init = True
        warnings.filterwarnings("ignore", "data discontinuity")

        rep = Reporter(gui_q)      # record() synthesizes zeros on a
        speaker_id, rec_ctx, rec = None, None, None   # silent device,
        backoff, last_check = 1.0, 0.0   # so this recheck always runs

        def close_rec():
            nonlocal rec_ctx, rec
            try:
                if rec_ctx is not None:
                    rec_ctx.__exit__(None, None, None)
            except Exception:
                pass
            rec_ctx = rec = None

        while not stop.is_set():
            try:
                now = time.time()
                if now - last_check > 2:
                    last_check = now
                    sp = sc.default_speaker()
                    if sp.id != speaker_id:
                        if speaker_id is not None:
                            diag(f"speaker changed -> {sp.name}")
                            rep.set("speaker", "warn", "reconnecting")
                        speaker_id = sp.id
                        close_rec()
                if rec is None:
                    mic = next((m for m in sc.all_microphones(
                        include_loopback=True)
                        if m.isloopback and m.id == speaker_id), None)
                    if mic is None:
                        rep.set("speaker", "err", "no loopback device")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 5)
                        continue
                    rec_ctx = mic.recorder(samplerate=SR, channels=1)
                    rec = rec_ctx.__enter__()
                    backoff = 1.0
                    rep.set("speaker", "ok")
                    diag("speaker capture ok")
                put_drop(qs, rec.record(512)[:, 0])
            except Exception as e:
                close_rec()
                rep.set("speaker", "err", f"capture failed: {e}")
                diag(f"loopback error: {e}; retry in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 5)
    except Exception as e:
        put_drop(qs, ("err", e))
    finally:
        try:
            close_rec()
        except Exception:
            pass
        if com_init:
            ctypes.windll.ole32.CoUninitialize()


# ------------------------------------------------------------- asr consumer

def asr_consumer(pipe, q, gui_q, log, log_lock, tag, trans_q=None,
                 idle_gap=1.0):
    """One stream, one consumer thread — mic can't starve behind
    speaker decode on CPU-only boxes. tag: 'spk' or 'mic'."""
    rep = Reporter(gui_q)
    last_p = {}
    stat_a = stat_d = 0.0                  # audio seconds, decode seconds
    stat_t = last_chunk = time.time()
    trans_seq = [0]
    who = "them" if tag == "spk" else "you"

    def emit(final):
        """One final, one fate: transcript log (with seg id so the
        translation lines match), translate, GUI, dedup reset."""
        seg = ""
        if trans_q is not None:
            trans_seq[0] += 1
            if put_drop(trans_q, (trans_seq[0], final)):
                with log_lock:             # eviction skips a seg too
                    log.write("    -> (trans_q full: seg dropped)\n")
            seg = f"  [seg {trans_seq[0]}]"
        with log_lock:
            log.write(f"[{time.strftime('%H:%M:%S')}] "
                      f"[{who}] {final}{seg}\n")
        put_drop(gui_q, ("final", tag, final))
        last_p.pop(tag, None)              # reset dedup for next utterance

    while not stop.is_set():
        try:
            chunk = q.get(timeout=0.02)
            last_chunk = time.time()
        except queue.Empty:
            # loopback may stop delivering on silence — feed zeros so
            # the endpoint rule still finalizes the utterance; the
            # returned final must be emitted, not dropped
            if pipe.partials.get(tag) \
                    and time.time() - last_chunk > idle_gap:
                try:
                    final = pipe.feed(np.zeros(1024, np.float32), tag)
                    if final:
                        emit(final)
                except Exception:
                    pass
            continue
        if isinstance(chunk, tuple):      # ("err", exc) from loopback
            put_drop(gui_q, ("err",
                             f"speaker capture failed: {chunk[1]}"))
            continue
        try:
            t0 = time.monotonic()
            final = pipe.feed(chunk, tag)
            stat_a += len(chunk) / SR
            stat_d += time.monotonic() - t0
            rep.set(f"asr-{tag}", "ok")
            if time.time() - stat_t > 5:
                diag(f"stat decode {tag}:rtf="
                     f"{stat_d/max(stat_a,0.01):.2f} q={q.qsize()}")
                stat_t, stat_a, stat_d = time.time(), 0.0, 0.0
            if final:
                emit(final)
            elif pipe.partials.get(tag):
                p = pipe.partials[tag]
                # raw zipformer partials are ALL-CAPS — sentence-case
                # them (leave cased models like nemotron alone)
                if p.isupper():
                    p = p.capitalize()
                if p != last_p.get(tag):   # dedupe identical partials
                    last_p[tag] = p
                    put_drop(gui_q, ("partial", tag, p))
        except Exception as e:
            rep.set(f"asr-{tag}", "err", f"decode err: {e}")
    final = pipe.flush(tag)                # emit the last utterance
    if final:
        emit(final)


# ------------------------------------------------------------------ overlay

BG, DIM, TXT, MUTE, ACC = "#101418", "#9ca3af", "#f9fafb", "#4b5563", "#93c5fd"
# compiled builds may live somewhere read-only (Program Files);
# logs go to %LOCALAPPDATA% there, repo logs/ in dev
if "__compiled__" in globals():
    LOGS = Path(os.environ.get("LOCALAPPDATA", ROOT)) / "livecap" / "logs"
else:
    LOGS = ROOT / "logs"
CFG = LOGS / "livecap-window.json"


class Overlay:
    def __init__(self, gui_q, status_items, mute, den_off,
                 mic_active=True):
        self.q = gui_q
        self.mute, self.den_off = mute, den_off
        self.mic_active = mic_active
        self.them = {"last": "", "now": ""}
        self.me = ""
        self.tr = ""
        self.finals = deque(maxlen=50)
        self.fs, self.alpha = 11, 0.85
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        r = self.root = tk.Tk()
        # fonts scale with DPI automatically; pixel constants don't —
        # scale them by the same factor (1.0 at 96 DPI)
        self.dpi = r.winfo_fpixels("1i") / 96
        r.overrideredirect(True)
        r.attributes("-topmost", True, "-alpha", self.alpha)
        r.configure(bg=BG)
        geo = self._load_geo()
        w, h = geo.get("w", int(560 * self.dpi)), \
            geo.get("h", int(100 * self.dpi))
        x = geo.get("x", (r.winfo_screenwidth() - w) // 2)
        y = geo.get("y", r.winfo_screenheight() - h - 90)
        # clamp into the virtual screen (all monitors, incl. negative
        # coords left/above primary) — saved geometry can be off-screen
        try:
            import ctypes
            u = ctypes.windll.user32
            vx, vy = u.GetSystemMetrics(76), u.GetSystemMetrics(77)
            vw, vh = u.GetSystemMetrics(78), u.GetSystemMetrics(79)
        except Exception:
            vx = vy = 0
            vw, vh = r.winfo_screenwidth(), r.winfo_screenheight()
        x = min(max(x, vx), vx + vw - w)
        y = min(max(y, vy), vy + vh - h)
        self._vtop = vy                # growth clamps against this
        r.geometry(f"{w}x{h}+{x}+{y}")
        pad = int(24 * self.dpi)
        mk = lambda **kw: tk.Label(r, bg=BG, anchor="w", justify="left",
                                   wraplength=w - pad, **kw)

        top = tk.Frame(r, bg=BG)
        top.pack(fill="x", pady=(0, 2))
        # buttons pack first so a wide badge row can never clip them
        if self.mic_active:
            self.mute_lbl = self._btn(top, "mic: on", self.tgl_mute)
            self.den_lbl = self._btn(top, "denoise: on", self.tgl_den)
        self._btn(top, "copy all", self.copy_all)
        self._btn(top, "copy", self.copy_last)
        self._btn(top, "×", self.close)
        self.badges = tk.Frame(top, bg=BG)
        self.badges.pack(side="left", padx=(10, 4), fill="x",
                         expand=True)
        self.dot = tk.Label(self.badges, text="●", fg="#16a34a",
                            bg=BG, font=("Segoe UI", 10))
        self.dot.pack(side="left", padx=(0, 6))
        self.state_lbls, self.state_lvl, self.state_txt = {}, {}, {}
        self.info_lbl = None
        for comp, lvl, txt in status_items:
            self._set_state(comp, lvl, txt)

        self.esc_lbl = mk(fg="#fbbf24", font=("Segoe UI", 9))
        self.them_last = mk(fg=DIM, font=("Segoe UI", 10))
        self.trans = mk(fg=ACC, font=("Segoe UI", self.fs - 3))
        self.them_now = mk(fg=TXT, font=("Segoe UI", 11))
        self.me_now = mk(fg=DIM, font=("Segoe UI", 10))
        self.text_lbls = (self.esc_lbl, self.them_last, self.trans,
                          self.them_now, self.me_now)
        for lbl in self.text_lbls:
            lbl.pack(fill="x", padx=12)
        grip = mk(fg=MUTE, font=("Segoe UI", 8), text="◢",
                  cursor="size_nw_se")
        grip.pack(anchor="se", padx=4)
        grip.bind("<Button-1>", self._rs_down)
        grip.bind("<B1-Motion>", self._rs_move)

        for wdg in (r, top, self.badges, *self.text_lbls):
            wdg.bind("<Button-1>", self._down)
            wdg.bind("<B1-Motion>", self._move)
        r.bind("<Escape>", self._esc)
        r.bind("m", lambda e: self.tgl_mute())
        r.bind("d", lambda e: self.tgl_den())
        r.bind("c", lambda e: self.copy_last())
        r.bind("<MouseWheel>", self._wheel)
        signal.signal(signal.SIGINT, self._sigint)
        r.update_idletasks()
        r.report_callback_exception = lambda t, v, tb: diag(
            f"tk callback: {v}")
        # WDA_EXCLUDEFROMCAPTURE: on screen normally, invisible to
        # screenshots and screen-share capture (Windows 10 2004+)
        hidden = False
        try:
            import ctypes
            hidden = bool(ctypes.windll.user32.SetWindowDisplayAffinity(
                int(r.wm_frame(), 16), 0x11))
        except Exception:
            pass
        if hidden:
            self._set_state("cap", "ok", "hidden-from-capture")
        else:
            # info, not warn — it isn't actionable, and an amber dot
            # all session would cry wolf
            self._set_state("cap", "info", "in captures")
        r.after(30, self._poll)

    LVLCOL = {"ok": "#14532d", "warn": "#78350f", "err": "#7f1d1d",
              "info": "#243244"}
    DOTCOL = {"ok": "#16a34a", "warn": "#d97706", "err": "#dc2626",
              "info": "#9ca3af"}
    _ORDER = {"err": 3, "warn": 2, "ok": 1, "info": 0}

    def _set_state(self, comp, lvl, text=""):
        """Typed per-component status — plan_badges() picks the row
        (one badge max + '+N', chip hidden under err); rebuilt on each
        event since badges are few and events are deduped."""
        self.state_lvl[comp] = lvl
        self.state_txt[comp] = text
        shown, extra, chip = plan_badges(self.state_lvl,
                                         self.state_txt)
        for lbl in self.state_lbls.values():
            lbl.destroy()
        self.state_lbls.clear()
        if self.info_lbl is not None:
            self.info_lbl.destroy()
            self.info_lbl = None
        for c in shown:
            l = self.state_lvl[c]
            icon = "⚠" if l == "warn" else "✗"
            txt = (self.state_txt.get(c) or icon)[:40]
            if not txt.startswith(icon):
                txt = f"{icon} {txt}"
            if extra:
                txt += f" +{extra}"
            lbl = tk.Label(self.badges, text=txt, fg="#e5e7eb",
                           bg=self.LVLCOL[l],
                           font=("Segoe UI", 8), padx=8, pady=2)
            lbl.pack(side="left", padx=(0, 4))
            # click-to-list: "+N" hides the rest — clicking shows all
            lbl.bind("<Button-1>", self._list_problems)
            self.state_lbls[c] = lbl
        if chip:
            self.info_lbl = tk.Label(
                self.badges, text=chip, fg="#e5e7eb",
                bg=self.LVLCOL["info"], font=("Segoe UI", 8),
                padx=8, pady=2)
            self.info_lbl.pack(side="left", padx=(0, 4))
        worst = max(self.state_lvl.values(), key=self._ORDER.get,
                    default="ok")
        self.dot.config(fg=self.DOTCOL.get(worst, "#16a34a"))

    def _btn(self, parent, text, fn):
        b = tk.Button(parent, text=text, fg=DIM, bg="#243244",
                      activebackground="#374151", activeforeground=TXT,
                      bd=0, relief="flat", cursor="hand2",
                      font=("Segoe UI", 9), padx=8, command=fn)
        b.pack(side="right", padx=3, pady=2)
        return b

    def _list_problems(self, _e=None):
        """Click a badge -> the hidden '+N' problems list briefly in
        the prompt line."""
        probs = [f"{c}: {self.state_txt.get(c) or l}"
                 for c, l in self.state_lvl.items()
                 if l in ("warn", "err")]
        if probs:
            self.esc_lbl.config(text=" · ".join(probs))
            self.root.after(3500, self._esc_clear)

    def _sigint(self, *_):
        # Tcl's event wait swallows Ctrl+C; after() wakes the loop
        try:
            self.root.after(0, self.close)
        except Exception:
            pass

    def _down(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), \
            e.y_root - self.root.winfo_y()

    def _move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _rs_down(self, e):
        self._rx, self._ry = e.x_root, e.y_root
        self._rw, self._rh = self.root.winfo_width(), self.root.winfo_height()

    def _rs_move(self, e):
        w = max(int(360 * self.dpi), self._rw + e.x_root - self._rx)
        h = max(int(60 * self.dpi), self._rh + e.y_root - self._ry)
        self.root.geometry(f"{w}x{h}")
        for lbl in self.text_lbls:
            lbl.config(wraplength=w - int(24 * self.dpi))

    def _wheel(self, e):
        if e.state & 0x4:                        # ctrl+wheel: font size
            self.fs = max(8, min(20, self.fs + (1 if e.delta > 0 else -1)))
            self.them_last.config(font=("Segoe UI", self.fs - 1))
            self.them_now.config(font=("Segoe UI", self.fs))
            self.trans.config(font=("Segoe UI", self.fs - 3))
            self.me_now.config(font=("Segoe UI", self.fs - 1))
        else:                                    # wheel: opacity
            self.alpha = max(0.35, min(1.0,
                                       self.alpha + (0.05 if e.delta > 0
                                                     else -0.05)))
            self.root.attributes("-alpha", self.alpha)

    def _esc(self, _e=None):
        now = time.time()
        if getattr(self, "_esc_armed", 0) > now:
            self.close()
            return
        self._esc_armed = now + 2.5
        self.esc_lbl.config(text="press Esc again to quit")
        self.root.after(2600, self._esc_clear)

    def _esc_clear(self):
        if getattr(self, "_esc_armed", 0) <= time.time():
            try:                           # window may already be gone
                self.esc_lbl.config(text="")
            except Exception:
                pass

    def tgl_mute(self):
        if not self.mic_active:
            return
        on = self.mute.is_set()                  # muted now? -> unmute
        (self.mute.clear if on else self.mute.set)()
        if self.mic_active:
            self.mute_lbl.config(text="mic: on" if on else "mic: muted")
        if not on:
            self._set_state("mute", "err", "MIC MUTED")
        else:
            self._set_state("mute", "ok")

    def tgl_den(self):
        if not self.mic_active:
            return
        on = self.den_off.is_set()               # denoise off? -> on
        (self.den_off.clear if on else self.den_off.set)()
        if self.mic_active:
            self.den_lbl.config(text="denoise: on" if on else "denoise: off")

    def copy_last(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.them["now"] or self.them["last"])

    def copy_all(self):
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(self.finals))

    def _load_geo(self):
        try:
            g = json.loads(CFG.read_text())
            return g if isinstance(g, dict) else {}
        except Exception:
            return {}

    def _poll(self):
        try:
            self.root.after(30, self._poll)   # reschedule first — a bad
        except tk.TclError:                   # event can't stop draining
            return
        if stop.is_set():
            return
        dirty = False
        try:
            while True:
                ev = self.q.get_nowait()
                dirty = True
                if ev[0] == "partial":
                    _, tag, text = ev
                    if tag == "spk":
                        self.them["now"] = text
                    else:
                        self.me = "you: " + text
                elif ev[0] == "final":
                    _, tag, text = ev
                    self.finals.append(
                        ("them" if tag == "spk" else "you") + ": " + text)
                    if tag == "spk":
                        self.them["last"], self.them["now"] = text, ""
                    else:
                        self.me = "you: " + text
                elif ev[0] == "trans":
                    self.tr = ev[1]
                elif ev[0] == "state":
                    # badge/dot only — never overwrite the caption
                    self._set_state(ev[1], ev[2],
                                    ev[3] if len(ev) > 3 else "")
                elif ev[0] == "err":
                    self.them["now"] = str(ev[1])
                    self._set_state("worker", "err", str(ev[1])[:40])
        except queue.Empty:
            pass
        except Exception:                    # one bad event must not
            pass                             # kill the drain loop
        if dirty:
            zwsp = "\u200b"
            self.them_last.config(text=self.them["last"] or zwsp)
            self.them_now.config(text=self.them["now"] or zwsp)
            self.trans.config(text=self.tr or zwsp)
            self.me_now.config(text=self.me or zwsp)
            # grow to fit wrapped text (never auto-shrinks) — anchored
            # at the bottom edge so growth can't leave the screen
            self.root.update_idletasks()
            rh = self.root.winfo_reqheight()
            h0 = self.root.winfo_height()
            if rh > h0 + 2:
                new_y = max(self._vtop,
                            self.root.winfo_y() + h0 - rh)
                self.root.geometry(f"{self.root.winfo_width()}x{rh}"
                                   f"+{self.root.winfo_x()}+{new_y}")

    def close(self):
        if getattr(self, "_closed", False):   # Esc x2 + Ctrl+C can
            return                            # both land here
        self._closed = True
        try:
            CFG.write_text(json.dumps({
                "x": self.root.winfo_x(), "y": self.root.winfo_y(),
                "w": self.root.winfo_width(), "h": self.root.winfo_height()}))
        except Exception:
            pass
        stop.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", metavar="CODE", default="ja",
                    help="translate [them] captions to this language "
                         "(e.g. ko, ja, vi, zh-CN; default: ja)")
    ap.add_argument("--log-days", metavar="N", type=int, default=0,
                    help="delete transcripts older than N days "
                         "(default: keep)")
    ap.add_argument("--trans-email", metavar="EMAIL",
                    default=os.environ.get("LIVECAP_TRANS_EMAIL"),
                    help="email for MyMemory — raises free quota ~10x "
                         "(or set LIVECAP_TRANS_EMAIL)")
    args = ap.parse_args()

    # open the diag log FIRST — every earlier startup line (ASR model,
    # provider, cable attempts) has nowhere to go in the windowed
    # build, so it must be capturing before anything else runs
    LOGS.mkdir(parents=True, exist_ok=True)
    dp = LOGS / "livecap-diag.log"
    if dp.exists() and dp.stat().st_size > 2 * 1024 * 1024:
        dp.replace(LOGS / "livecap-diag.1.log")
    _diag_path[0] = dp
    _diag_size[0] = dp.stat().st_size if dp.exists() else 0
    _diag_log[0] = open(dp, "a", encoding="utf-8", buffering=1)
    sys.excepthook = lambda t, v, tb: diag_exc("fatal", t, v, tb)

    # preflight BEFORE any sherpa call — a missing file exits the
    # process silently from native code when the console is hidden
    model_dir = find_asr_dir()
    missing = preflight(ROOT, model_dir)
    if missing:
        raise SystemExit("missing model files:\n  "
                         + "\n  ".join(str(m) for m in missing))

    den = make_denoiser()
    diag(f"ASR model: {model_dir.name}")
    rec, asr_provider = make_recognizer(model_dir)
    diag(f"ASR provider: {asr_provider}")
    punct = None if "nemotron" in model_dir.name.lower() else make_punctuator()
    pipe = Pipeline(rec, punct)

    status = []
    # queues first — the cable's finished_callback reports to gui_q
    mic_q = queue.Queue(maxsize=10)      # voice path: <160ms, drop not delay
    micasr_q = queue.Queue(maxsize=64)   # ~1s of 16ms chunks
    qs = queue.Queue(maxsize=64)         # ~2s of 32ms chunks
    gui_q = queue.Queue(maxsize=1024)     # bounded — a stalled overlay
                                          # can't eat RAM
    ring, out_stream, cable_sr = None, None, SR
    cands = cable_candidates()
    if not cands:
        status.append(("cable","warn","no VB-CABLE"))
        diag("VB-CABLE not found — captions only, no mic passthrough")
    elif not cable_usable():
        status.append(("cable","warn","cable hidden"))
        diag("cable not visible to apps (session) — mic capture off")
    else:
        def _open_cable(idx, sr, wasapi, ring):
            kw = {"latency": "low"}
            if wasapi:                    # mono on stereo shared mode
                kw["extra_settings"] = sd.WasapiSettings(
                    auto_convert=True)
            return sd.OutputStream(
                device=idx, samplerate=sr, channels=1,
                dtype="float32",
                callback=lambda data, frames, t, s:
                    data.__setitem__((slice(None), 0),
                                    ring.pull(frames)),
                finished_callback=lambda: put_drop(
                    gui_q, ("state", "cable", "err",
                            "cable stream stopped")), **kw)
        ring, out_stream, cable_sr = open_cable(
            cands, True, _open_cable)
        if out_stream is None:
            status.append(("cable","err","cable open failed"))
        else:
            status.append(("cable","ok","cable ✓"))
            diag(f"cable latency: {out_stream.latency}")

    mute, den_off = threading.Event(), threading.Event()
    log_lock = threading.Lock()
    # log retention: sweep transcripts older than --log-days (0 = keep)
    if getattr(args, "log_days", 0):
        cutoff = time.time() - args.log_days * 86400
        for f in LOGS.glob("livecap-*.txt"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    log = open(LOGS / f"livecap-{time.strftime('%Y%m%d-%H%M%S')}.txt",
               "w", encoding="utf-8", buffering=1)
    threading.excepthook = lambda a: (
        diag_exc(f"thread {a.thread.name} died",
                 a.exc_type, a.exc_value, a.exc_traceback),
        put_drop(gui_q, ("state", "worker", "err", f"{a.thread.name} died")))
    trans_q = queue.Queue(maxsize=4) if args.lang else None

    in_stream, mic_sr = None, SR
    if out_stream is None:
        status.append(("mic","warn","mic off"))
    else:
        mic_idx = pick_mic()
        if mic_idx is None:
            status.append(("mic","warn","no mic"))
        else:
         try:
            st, mic_sr = open_mic(mic_idx, den, mic_q, gui_q)
            st.start()
            in_stream = st                 # assign only after start()
            status.append(("mic","ok","mic ✓"))
            diag(f"mic open: dev{mic_idx} {mic_sr}Hz "
                 f"latency={st.latency}")
         except Exception as e:
            status.append(("mic","warn","no mic"))
            diag(f"no mic input ({e})")
    if out_stream is not None:
        try:
            out_stream.start()
        except Exception as e:         # degrade to captions-only,
            diag(f"cable start failed: {e}")   # don't die at startup
            try:
                out_stream.close()
            except Exception:
                pass
            out_stream = ring = None
            status.append(("cable","err","start failed"))
    workers = []
    worker_defs = []
    if in_stream is not None:
        # no mic stream -> no chunks ever -> the stall detector would
        # false-positive; the engine has nothing to do anyway
        worker_defs.append(
            (voice_engine, "voice",
             dict(args=(mic_q, den, ring, micasr_q, mute, den_off,
                        gui_q),
                  kwargs=dict(mic_sr=mic_sr, cable_sr=cable_sr))))
    worker_defs += [
            (loopback_pump, "spk-pump", dict(args=(qs, gui_q))),
            (asr_consumer, "asr-spk",
             dict(args=(pipe, qs, gui_q, log, log_lock,
                        "spk", trans_q))),
            (asr_consumer, "asr-mic",
             dict(args=(pipe, micasr_q, gui_q, log,
                        log_lock, "mic"))),
    ]
    for tgt, name, kw in worker_defs:
        t = threading.Thread(target=tgt, daemon=True, name=name, **kw)
        t.start()
        workers.append(t)
    if trans_q is not None:
        # honest disclosure: captions leave the box for translation
        status.append(("trans","info",f"{args.lang}·online"))
        t = threading.Thread(target=translator_worker, daemon=True,
                             name="trans",
                             args=(trans_q, gui_q, log, log_lock,
                                   lang_code(args.lang),
                                   args.trans_email))
        t.start()
        workers.append(t)

    status.append(("log","info","log"))  # model name lives in diag
    diag("livecap running — mic -> denoise -> cable, captions for both")
    try:
        Overlay(gui_q, status, mute, den_off,
                mic_active=in_stream is not None).run()
    finally:
        stop.set()
        for w in workers:                # let workers flush + emit
            w.join(timeout=3)
            if w.is_alive():
                diag(f"shutdown: {w.name} still blocked after 3s")
        try:
            log.close()
        except Exception:
            pass
        for st in (out_stream, in_stream):    # each op guarded — a
            if st is None:                    # failed stop() must not
                continue                      # skip the other stream
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass
        diag("shutdown complete")
        diag_flush()                   # the last lines must land
        try:
            if _diag_log[0] is not None:
                _diag_log[0].close()
        except Exception:
            pass
        _diag_log[0] = None


def _fatal_box(msg):
    """Console may be hidden in the built exe — surface fatal startup
    failures as a dialog, not silence."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "livecap", 0x10)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # SystemExit(str) is our own clean startup error — the
        # windowed build has no console, so it needs the dialog too
        if isinstance(e.code, str):
            _fatal_box(f"livecap failed to start:\n{e.code}")
        diag_flush()               # fatal lines must hit the file
        raise
    except Exception as e:
        _fatal_box(f"livecap failed to start:\n{e}")
        diag_exc("fatal", type(e), e, e.__traceback__)
        diag_flush()
        raise
