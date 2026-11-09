#!/usr/bin/env python
"""livecap — Krisp-style live captions + mic denoise passthrough.

  mic (16k) -> GTCRN denoise -> VB-CABLE virtual mic (what call apps hear)
           -> [you] caption (raw mic — denoise can eat consonants)
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
SR = 16000

stop = threading.Event()

_single = []                                 # keeps the handle alive
_splash = [None]


def _dpi_aware():
    """System DPI awareness (mode 1) — must run BEFORE any Tk window
    exists or it renders blurry. Tk 8.6 can't track per-monitor
    changes anyway."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def run_with_pump(fn, pump, period=0.05):
    """Run fn on a worker while the main thread pumps the Tk event
    loop — a frozen window ghosts as 'Not responding' in seconds."""
    out = {}
    def w():
        try:
            out["v"] = fn()
        except BaseException as e:
            out["e"] = e
    t = threading.Thread(target=w, daemon=True)
    t.start()
    while t.is_alive():
        pump()
        t.join(period)
    if "e" in out:
        raise out["e"]
    return out["v"]


def acquire_single_instance() -> bool:
    """Named-mutex single instance — two copies push two streams into
    the same cable and the call hears the voice twice, delayed.
    Returns False when another instance holds the mutex; any API
    failure fails open (never blocks startup)."""
    try:
        import ctypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        _single.append(
            k.CreateMutexW(None, False, "Local\\livecap-singleton"))
        return ctypes.get_last_error() != 183   # ERROR_ALREADY_EXISTS
    except Exception:
        return True


def loading_splash():
    """Immediate feedback while models load — several seconds on slow
    CPUs with the console hidden. Returns the window or None."""
    try:
        sp = tk.Tk()
        sp.overrideredirect(True)
        sp.attributes("-topmost", True)
        sp.configure(bg="#101418")
        w = int(280 * sp.winfo_fpixels("1i") / 96)
        h = int(56 * sp.winfo_fpixels("1i") / 96)
        x = (sp.winfo_screenwidth() - w) // 2
        y = sp.winfo_screenheight() - h - 90
        sp.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(sp, text="livecap — loading models…",
                 fg="#f9fafb", bg="#101418",
                 font=("Segoe UI", 10)).pack(expand=True)
        sp.update()
        _splash[0] = sp
        return sp
    except Exception:
        return None


def drop_splash():
    """Destroy the loading splash — called on success AND on every
    fatal path so it can't linger topmost behind the error box."""
    sp, _splash[0] = _splash[0], None
    if sp is not None:
        try:
            sp.destroy()
        except Exception:
            pass


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
            prop = max(-0.001, min(0.001,      # damping: zeta ~0.6
                       (err / sr) * 0.1))      # (capped +-0.1%)
        else:                                  # old proportional gain
            prop = 0.0
        self.trim = max(-0.001, min(0.001, self.trim))
        self.step = self.base_step * (1 + self.trim + prop)


def asr_dir() -> Path:
    # nemotron-560ms ONLY — feature_dim=128 is baked in, so falling
    # back to another model would decode garbage
    models = [d for d in (ROOT / "models").iterdir()
              if (d / "tokens.txt").is_file()
              and "nemotron" in d.name and "560ms" in d.name]
    if not models:
        raise SystemExit("no nemotron-560ms model dir under models/")
    return sorted(models)[0]


ASR_TAR_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/"
               "download/asr-models/sherpa-onnx-nemotron-speech-"
               "streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2")


def ensure_asr_model():
    """Fetch nemotron-560ms into models/ if absent — runs on the
    loader thread with the splash up (the tar is ~250MB packed)."""
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    if [d for d in models.iterdir()
            if "nemotron" in d.name and "560ms" in d.name]:
        return
    import tarfile, urllib.request
    dst = models / "nemotron-dl.tar.bz2"
    diag("downloading nemotron-560ms model...")
    urllib.request.urlretrieve(ASR_TAR_URL, dst)
    with tarfile.open(dst, "r:bz2") as t:
        t.extractall(models)
    dst.unlink()
    diag("nemotron model unpacked")


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
              sample_rate=SR, feature_dim=128,   # nemotron 128 mel bins
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
    cfg.model.num_threads = 1        # GTCRN is tiny — intra-op
                                     # threads just spin-wait and
                                     # starve the ASR of real cores
    return sherpa_onnx.OnlineSpeechDenoiser(cfg)



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
    ('ok') clears badges the error path set. Share ONE instance: the
    dedupe memory is what makes a recovery clear another component's
    error."""

    def __init__(self, q):
        self.q, self.last = q, {}
        self._lock = threading.Lock()

    def set(self, comp, lvl, msg=""):
        with self._lock:
            if self.last.get(comp) != (lvl, msg):
                self.last[comp] = (lvl, msg)
                put_drop(self.q, ("state", comp, lvl, msg))


def badge_shown(lvl: str) -> bool:
    """Only warn/err get badges — ok folds into the dot, info into the
    single compact chip, so the status row stays inside the window."""
    return lvl in ("warn", "err")


_BADGE_PRIORITY = ("mute", "cable", "mic", "speaker")


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
        put_drop(gui_q, ("state", "trans", "warn", "translation off"))
        diag(f"translator import failed: {e}")
        return
    gmap = {"zh-CN": "zh-CN", "zh-TW": "zh-TW", "zh-HK": "zh-TW",
            "nb-NO": "no", "he-IL": "iw"}   # he/zh-HK rejected by Google
    gcode = gmap.get(lang, lang.split("-")[0])
    try:
        tr = GoogleTranslator(source="en", target=gcode)
    except Exception as e:
        tr = None
        put_drop(gui_q, ("state", "trans", "warn", "google off"))
    try:
        fallback = MyMemoryTranslator(source="en-US", target=lang,
                                      email=email)
    except Exception as e:
        fallback = None
        put_drop(gui_q, ("state", "trans", "warn", "mymemory off"))
    if tr is None and fallback is None:
        put_drop(gui_q, ("state", "trans", "warn", "translation off"))
        return
    warned = False
    while not stop.is_set():
        try:
            sid, text = tq.get(timeout=0.1)
        except queue.Empty:
            continue
        # scrollback keeps every segment — a skipped one is a
        # permanent gap, so only skip under a real backlog
        while tq.qsize() > 3:
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
            put_drop(gui_q, ("trans", out, sid))
            with log_lock:
                log.write(f"    -> {out}  (seg {sid})\n")
        elif not warned:                   # visible once, not per call
            warned = True
            put_drop(gui_q, ("trans",
                             f"[translation paused: {reason or 'unknown'}]",
                             sid))


# ------------------------------------------------------------------- asr

class Pipeline:
    def __init__(self, recognizer):
        self.rec = recognizer
        self.streams = {}
        self.partials = {}
        self.lock = threading.Lock()   # serialize decode across streams

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
            ran = False                  # get_result builds a string
            while self.rec.is_ready(stream):   # — only pay for it
                self.rec.decode_stream(stream)  # when decode ran or
                ran = True                      # an endpoint fired
            if self.rec.is_endpoint(stream):
                text = result_text(
                    self.rec.get_result(stream)).strip()
                self.rec.reset(stream)
                self.partials.pop(tag, None)
            elif ran:
                self.partials[tag] = result_text(
                    self.rec.get_result(stream)).strip()
        return text or None

    def flush(self, tag):
        """Force-finalize one stream (per consumer); returns pending
        text or None. Padding silence is required — streaming
        transducers drop final tokens without it."""
        with self.lock:                  # both consumers flush at exit
            stream = self.streams.get(tag)
            if stream is None:
                return None
            pad = np.zeros(int(1.0 * SR), dtype=np.float32)
            stream.accept_waveform(SR, pad)
            stream.input_finished()
            while self.rec.is_ready(stream):
                self.rec.decode_stream(stream)
            text = result_text(self.rec.get_result(stream)).strip()
            self.streams.pop(tag, None)
        return text or None


# ------------------------------------------------------ mic -> denoise ->
#                                                cable + [you] caption tap

class Ring:
    """Float32 ring buffer feeding the cable output callback."""

    def __init__(self, n: int):
        self.b = np.zeros(n, np.float32)
        self.n = n
        self.r = self.w = 0
        self.underruns = self.overruns = self.trims = 0
        self._armed = False              # underruns count only once
        self.lock = threading.Lock()     # real audio has flowed
        self.max_gap = self._last = 0.0  # callback-cadence metric

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
            now = time.monotonic()       # Q2 instrumentation: track
            if self._last:               # pull cadence — a GIL stall
                self.max_gap = max(self.max_gap, now - self._last)
            self._last = now             # shows up as a late callback
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
            self._armed = False
        if prefill:
            self.push(np.zeros(prefill, np.float32))
            self._armed = False            # prefill isn't real audio —
                                           # don't count its drains

    def drop_to(self, keep: int) -> int:
        """Discard all but the newest `keep` samples — a latency burst
        drains at 0.1%/s through the rate trim (minutes); skipping
        ~100ms of voice once is far cheaper."""
        with self.lock:
            extra = (self.w - self.r) - keep
            if extra <= 0:
                return 0
            self.r += extra
            self.trims += 1
            return extra


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


def cable_usable_threaded() -> bool:
    """soundcard needs COM on the calling thread — the supervisor
    thread has none. Init, check, uninit."""
    import ctypes
    ci = False
    try:
        ctypes.windll.ole32.CoInitialize(None)
        ci = True
    except Exception:
        pass
    try:
        return cable_usable()
    finally:
        if ci:
            ctypes.windll.ole32.CoUninitialize()


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


def open_mic(mic_idx, den, mic_q, gui_q, dead=None):
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
    stopped = dead or (lambda: put_drop(
        gui_q, ("state", "mic", "err", "mic stream stopped")))
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


def mark_dead(state, comp, rep, msg):
    """A stream died — flag it for the supervisor and show the badge."""
    if not state.get(comp + "_dead"):          # transition only —
        state[comp + "_dead"] = True           # don't spin the wait
        state["ev"].set()
    rep.set(comp, "err", msg)


def dead_cb(state, comp, rep, msg):
    """finished_callback for ONE stream generation. PortAudio fires it
    on stop()/close() too, so a replaced stream must not flag its
    successor — each new open bumps the generation."""
    state[comp + "_gen"] = gen = state.get(comp + "_gen", 0) + 1

    def cb():
        if state.get(comp + "_gen") == gen:
            mark_dead(state, comp, rep, msg)
    return cb


def _sd_reinit(state, rep):
    """PortAudio's device list is frozen at Pa_Initialize — a reinit
    needs every stream closed first. Dropping the live stream costs a
    ~1s passthrough gap, but a replugged device is INVISIBLE without
    it (frozen list returns stale indices forever). Whatever we close
    is marked dead so the supervisor reopens it next pass."""
    now = time.time()
    if now - state.get("reinit_t", 0.0) < 30.0:
        return False                 # one global cooldown: mic and cable
    state["reinit_t"] = now          # failures can't bounce each other
    live = {"in": state.get("in"), "out": state.get("out")}
    if live["out"] is not None:
        state["out"], state["ring"] = None, None
    if live["in"] is not None:
        state["in"] = None
    for st in live.values():
        if st is not None:
            try:
                st.stop(); st.close()
            except Exception:
                pass
    if live["out"] is not None:
        mark_dead(state, "cable", rep, "cable closed for reinit")
    if live["in"] is not None:
        mark_dead(state, "mic", rep, "mic closed for reinit")
    try:
        sd._terminate()
        sd._initialize()
        # indices shift on re-enumeration — a stale candidate list
        # would reopen the cable on the wrong device (or fail)
        state["cands"] = cable_candidates()
        diag("PortAudio reinitialized for new devices")
        return True
    except Exception as e:
        diag(f"sd reinit failed: {e}")
        return False


def stream_supervisor(state, ev, gui_q, den, mic_q,
                      open_mic_fn=None, open_cable_fn=None,
                      pick_fn=None, base=1.0, rep=None):
    """Reopens dead mic/cable streams with backoff (1,2,5s cap) —
    a USB unplug or driver reset shouldn't need an app restart.
    On success the engine sees the new ring/rates via `state`."""
    rep = rep or Reporter(gui_q)
    pick_fn = pick_fn or pick_mic
    open_mic_fn = open_mic_fn or (lambda i: open_mic(
        i, den, mic_q, gui_q,
        dead=dead_cb(state, "mic", rep, "mic stream stopped")))
    if open_cable_fn is None:
        def open_cable_fn():
            def fn(idx, sr, wasapi, ring):
                kw = {"latency": "low"}
                if wasapi:
                    kw["extra_settings"] = sd.WasapiSettings(
                        auto_convert=True)
                return sd.OutputStream(
                    device=idx, samplerate=sr, channels=1,
                    dtype="float32",
                    callback=lambda data, frames, t, s:
                        data.__setitem__((slice(None), 0),
                                        ring.pull(frames)),
                    finished_callback=dead_cb(
                        state, "cable", rep, "cable stream stopped"),
                    **kw)
            return open_cable(state["cands"], True, fn)
    backoff = {"mic": base, "cable": base}
    next_try = {"mic": 0.0, "cable": 0.0}
    fails = {"mic": 0, "cable": 0}
    next_scan = 0.0                      # cable rediscovery cadence
    scan_wait = 10.0                     # backs off on failed discovery
    while not stop.is_set():
        wait = 0.5                      # wake early for a pending retry
        for comp in ("mic", "cable"):
            if state[comp + "_dead"]:
                wait = min(wait, max(0.01,
                                     next_try[comp] - time.time()))
        ev.wait(wait)
        ev.clear()
        now = time.time()
        # cable installed/enabled mid-session: rescan on a cadence —
        # it only becomes visible after a PortAudio reinit (which the
        # 30s cooldown bounds); costs the mic ~1s once
        if state.get("cable_absent") and now >= next_scan:
            next_scan = now + scan_wait
            if cable_usable_threaded() and _sd_reinit(state, rep):
                if state["cands"]:
                    state["cable_absent"] = False
                    state["cable_dead"] = True     # existing reopen path
                    next_try["cable"] = 0.0
                    ev.set()
                    diag("cable discovered")
                else:                      # WASAPI sees it but
                    scan_wait = min(       # PortAudio doesn't — each
                        scan_wait * 2,     # failed discovery bounces
                        120)               # the mic ~1s; slow down
                    next_scan = now + scan_wait   # apply it NOW, not
                                                  # from the next pass
        if state["mic_dead"] and now >= next_try["mic"]:
            old = state["in"]
            if old is not None:                # drop the dead stream
                state["in"] = None             # NOW — its stale device
                try:                           # index blocks _sd_reinit
                    old.stop(); old.close()    # and replug detection
                except Exception:
                    pass
            try:
                idx = pick_fn()
                if idx is None and _sd_reinit(state, rep):
                    idx = pick_fn()    # re-enumerate for replugs
                if idx is None:
                    raise RuntimeError("no mic")
                st, msr = open_mic_fn(idx)
                st.start()
                state["in"] = st
                state["mic_sr"], state["mic_dead"] = msr, False
                state["mic_t"] = time.time()   # grace before stall
                backoff["mic"] = base
                fails["mic"] = 0
                rep.set("mic", "ok")
                diag(f"mic reopened: dev{idx} {msr}Hz")
            except Exception as e:
                fails["mic"] += 1
                if fails["mic"] >= 3:
                    _sd_reinit(state, rep)
                next_try["mic"] = now + backoff["mic"]
                if _should_log_fail(fails["mic"]):
                    diag(f"mic reopen failed ({e}); "
                         f"retry in {backoff['mic']:.0f}s")
                backoff["mic"] = min(backoff["mic"] * 2, 5 * base)
        if state["cable_dead"] and now >= next_try["cable"]:
            old = state["out"]
            if old is not None:                # drop the dead stream
                state["out"], state["ring"] = None, None   # NOW — a
                try:                           # stale object blocks
                    old.stop(); old.close()    # _sd_reinit; the engine
                except Exception:              # skips a None ring and
                    pass                       # rebuilds rs on reopen
            try:
                ring2, st2, csr = open_cable_fn()
                if st2 is None:
                    raise RuntimeError("all cable candidates failed")
                st2.start()
                state["out"] = st2
                state["ring"], state["cable_sr"] = ring2, csr
                state["cable_dead"] = False
                state["cable_t"] = time.time()
                backoff["cable"] = base
                fails["cable"] = 0
                rep.set("cable", "ok")
                diag(f"cable reopened: {csr}Hz")
            except Exception as e:
                fails["cable"] += 1
                if fails["cable"] >= 3:
                    _sd_reinit(state, rep)
                next_try["cable"] = now + backoff["cable"]
                if _should_log_fail(fails["cable"]):
                    diag(f"cable reopen failed ({e}); "
                         f"retry in {backoff['cable']:.0f}s")
                backoff["cable"] = min(backoff["cable"] * 2, 5 * base)


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
            f"trims={ring.trims if ring else 0} "
            f"gap={int(ring.max_gap * 1000) if ring else 0}ms "
            f"mic_q={mic_q.qsize()} drop={getattr(mic_q, 'drops', 0)} "
            f"micasr={micasr_q.qsize()} mdrop={getattr(micasr_q, 'drops', 0)} "
            f"denoise={avg_ms:.1f}ms max={max_ms:.1f}ms "
            f"slow={slow} fails={fails}")


def voice_engine(mic_q, den, ring, micasr_q, mute, den_off, gui_q,
                 mic_sr=SR, cable_sr=SR, stall_after=1.5, state=None,
                 rep=None):
    """mic -> denoise -> ring buffer for cable; raw chunk -> ASR queue.
    `state` (optional shared dict) lets the stream supervisor swap
    ring/rates on reopen — the engine picks them up per chunk."""
    rep = rep or Reporter(gui_q)             # dedupes; ok clears badges
    needs_reset = muted = False
    mic_rs = Resampler(mic_sr, SR) if mic_sr != SR else None
    pend = np.empty(0, np.float32)           # native-rate accumulator
    rs = Resampler(SR, cable_sr) if ring is not None else None
    target = int(0.04 * cable_sr)            # ~40ms target ring fill
    cur_sr = (mic_sr, cable_sr)
    cur_ring = ring
    shift = den.frame_shift_in_samples
    if ring is not None:                     # drop stale startup
        ring.reset(target)                   # chunks, not play them
    stat_t, stat_dn, stat_ms = time.time(), 0, 0.0
    stat_max = stat_slow = den_fails = 0
    last_chunk = time.time()
    seen_mt = 0                        # mic_t generation marker
    try:                                   # Pro Audio MMCSS class —
        import ctypes                      # priority scheduling for
        idx = ctypes.c_ulong()             # the 16ms deadline path
        ctypes.windll.avrt.AvSetMmThreadCharacteristicsW(
            "Pro Audio", ctypes.byref(idx))
    except Exception:
        pass
    while not stop.is_set():
        if state is not None:                # supervisor may have
            ring = state["ring"]             # swapped these on reopen
            new_sr = state["mic_sr"], state["cable_sr"]
            # mic rate ONLY — a cable reopen at a new rate must not
            # break the input resampler; pend's leftover partial frame
            # is at the old rate too
            if new_sr[0] != cur_sr[0]:
                mic_rs = Resampler(new_sr[0], SR) \
                    if new_sr[0] != SR else None
                pend = np.empty(0, np.float32)
            # rebuild rs on a ring OBJECT swap too, not just a rate
            # change — a reopened cable's clock skew differs from what
            # rs.trim learned; ring->None drops rs so the stat path
            # never calls ring.fill() on a missing ring
            if new_sr != cur_sr or ring is not cur_ring \
                    or (ring is not None and rs is None):
                cur_sr, cur_ring = new_sr, ring
                rs = Resampler(SR, cur_sr[1]) \
                    if ring is not None else None
                target = int(0.04 * cur_sr[1])
            mt = state.get("mic_t", 0)
            if mt != seen_mt:          # a NEW mic stream, not a stall —
                seen_mt = mt           # drop chunks the dead device left
                try:
                    while True:
                        mic_q.get_nowait()
                except queue.Empty:
                    pass
                pend = np.empty(0, np.float32)
                mic_rs = Resampler(cur_sr[0], SR) \
                    if cur_sr[0] != SR else None
                needs_reset = True     # den.reset() before next frame
                last_chunk = time.time()
        try:
            x = mic_q.get(timeout=0.02)
            last_chunk = time.time()
            rep.set("mic", "ok")             # Q12: recovery clears
        except queue.Empty:                  # the badge even muted
            # chunks keep arriving even while muted — a stall means
            # the device is gone, not silent
            grace = (state or {}).get("mic_t", 0)
            if time.time() - max(last_chunk, grace) > stall_after:
                rep.set("mic", "err", "no mic input")
                if state is not None and not state.get("mic_dead"):
                    state["mic_dead"] = True    # transition only —
                    state["ev"].set()           # don't spin the wait
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
            # [you] captions tap the RAW chunk, not the denoised one —
            # GTCRN can eat consonants, and captions shouldn't depend
            # on denoise output at all
            put_drop(micasr_q, x)
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
                # latency burst? trim can't drain it in reasonable
                # time — skip to the target and keep the newest audio
                if fill > target + int(0.06 * cur_sr[1]):
                    ring.drop_to(target)
                    fill = ring.fill()         # post-trim, not stale
                rs.steer(fill - target, cur_sr[1])
                ring.push(rs.resample(y))
            if time.time() - stat_t > 5:
                if rs is not None and ring is not None:
                    # saturated AND still off-target = real skew;
                    # trim alone at the cap is just windup
                    if abs(rs.trim) > 0.0009 and abs(
                            ring.fill() - target) > 0.01 * cur_sr[1]:
                        rep.set("cable", "err", "clock skew saturating")
                    else:
                        rep.set("cable", "ok")
                diag(voice_stat(ring, mic_q, micasr_q,
                                stat_ms / max(stat_dn, 1), stat_max,
                                stat_slow, den_fails))
                if ring is not None:
                    ring.max_gap = 0.0       # per-window metric
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
                put_drop(qs, rec.record(1600)[:, 0])   # 100ms chunks —
                # the model decodes in 560ms blocks anyway; 6x fewer
                # Python round-trips through the pump
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


# ----------------------------------------------------------------- hotkeys

def hotkey_worker(gui_q, tid_out=None):
    """Global hotkeys — tk bind() only fires while the overlay has
    focus, which mid-call it never does. RegisterHotKey(NULL hwnd)
    posts WM_HOTKEY to THIS thread's queue; a GetMessage loop drains
    it and pushes ("hotkey", action) into gui_q for _poll. Any
    failure is a badge + diag, never fatal."""
    WM_HOTKEY = 0x0312
    CTRL_ALT_NOREPEAT = 0x0002 | 0x0001 | 0x4000   # MOD_CONTROL |
    # MOD_ALT | MOD_NOREPEAT — held key can't toggle repeatedly.
    # Ctrl+Alt avoids call apps' own mute chords (Meet Ctrl+D,
    # Teams/Discord Ctrl+Shift+M, Zoom Alt+A).
    # id -> (action, key letter, modifiers, vk)
    HK = {1: ("mute", "m", CTRL_ALT_NOREPEAT, ord("M")),
          2: ("den",  "d", CTRL_ALT_NOREPEAT, ord("D"))}
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        if tid_out is not None:
            tid_out.append(
                ctypes.windll.kernel32.GetCurrentThreadId())
        registered = []
        for hkid, (act, letter, mod, vk) in HK.items():
            if u.RegisterHotKey(None, hkid, mod, vk):
                registered.append(hkid)
            else:
                put_drop(gui_q, ("state", "hotkey", "warn",
                                 f"ctrl+alt+{letter} taken"))
                diag(f"hotkey {act} reg failed: "
                     f"err {ctypes.get_last_error()}")
        if not registered:
            return
        # quiet chip so the chords are discoverable in-session
        put_drop(gui_q, ("state", "hk", "info",
                         "·".join(f"C-A-{HK[i][1]}"
                                  for i in registered)))
        diag("hotkeys registered: "
             + ", ".join(HK[i][0] for i in registered))
        try:
            msg = wintypes.MSG()
            while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    hk = HK.get(msg.wParam)
                    if hk:
                        put_drop(gui_q, ("hotkey", hk[0]))
        finally:
            for hkid in registered:
                u.UnregisterHotKey(None, hkid)
    except Exception as e:
        diag(f"hotkey worker failed: {e}")


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
    thr = 0.002 if tag == "spk" else 0.008   # mic has a noise floor
    hang = 0.0                       # silent seconds still fed —
                                     # > rule2 so endpoints still fire

    def emit(final):
        """One final, one fate: transcript log (with seg id so the
        translation lines match), translate, GUI, dedup reset. The
        final reaches the GUI before the translation is queued, so a
        translation can never beat its own caption to the overlay."""
        seq = 0
        if trans_q is not None:
            trans_seq[0] += 1
            seq = trans_seq[0]
        with log_lock:
            log.write(f"[{time.strftime('%H:%M:%S')}] [{who}] {final}"
                      + (f"  [seg {seq}]" if seq else "") + "\n")
        put_drop(gui_q, ("final", tag, final, seq))
        if trans_q is not None and put_drop(trans_q, (seq, final)):
            with log_lock:                 # eviction skips a seg too
                log.write("    -> (trans_q full: seg dropped)\n")
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
        # energy gate: call sides are silent ~half the time — don't
        # decode pure-zero loopback / idle mic; the idle_gap path
        # above still force-finalizes a pending partial
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > thr:
            hang = 1.5
        elif hang > 0:
            hang -= len(chunk) / SR
        if hang <= 0 and not pipe.partials.get(tag):
            continue
        try:
            t0 = time.monotonic()
            final = pipe.feed(chunk, tag)
            stat_a += len(chunk) / SR
            stat_d += time.monotonic() - t0
            rep.set(f"asr-{tag}", "ok")
            if time.time() - stat_t > 5:
                diag(f"stat decode {tag}:rtf="
                     f"{stat_d/max(stat_a,0.01):.2f} q={q.qsize()} "
                     f"drop={getattr(q, 'drops', 0)}"
                     + (f" tdrop={getattr(trans_q, 'drops', 0)}"
                        if trans_q is not None else ""))
                stat_t, stat_a, stat_d = time.time(), 0.0, 0.0
            if final:
                emit(final)
            elif pipe.partials.get(tag):
                p = pipe.partials[tag]
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


class CaptionView:
    """Append-only scrollback. Finals are permanent lines; the live
    partial is one tail region. Words two consecutive partials agree
    on are 'stable' (bright); the revising tail is dim — a word you
    already read never changes underneath you."""

    def __init__(self, parent, lines=6, fs=11):
        t = self.t = tk.Text(
            parent, bg=BG, fg=TXT, bd=0, highlightthickness=0,
            wrap="word", height=lines, font=("Segoe UI", fs),
            padx=12, pady=2, spacing3=3, cursor="arrow",
            state="disabled")
        t.tag_configure("stable", foreground=TXT)
        t.tag_configure("tail", foreground=DIM)
        t.tag_configure("tr", foreground=ACC,
                        font=("Segoe UI", fs - 3))
        t.mark_set("pstart", "end-1c")
        t.mark_gravity("pstart", "left")
        self.prev = []

    def _edit(self, fn):
        # only follow the live edge if already at the bottom —
        # scrolling up to re-read shouldn't be yanked back down
        at_bottom = self.t.yview()[1] >= 0.999
        self.t.configure(state="normal")
        fn()
        self.t.configure(state="disabled")
        if at_bottom:
            self.t.see("end")

    def partial(self, text):
        w = text.split()
        k = 0
        while (k < min(len(w), len(self.prev))
               and w[k] == self.prev[k]):
            k += 1
        self.prev = w

        def go():
            self.t.delete("pstart", "end-1c")
            if k:
                self.t.insert("end-1c", " ".join(w[:k]) + " ", "stable")
            if k < len(w):
                self.t.insert("end-1c", " ".join(w[k:]), "tail")
        self._edit(go)

    def final(self, text, sid):
        self.prev = []

        def go():
            self.t.delete("pstart", "end-1c")
            self.t.insert("end-1c", text, "stable")
            self.t.mark_set(f"seg{sid}", "end-1c")
            self.t.mark_gravity(f"seg{sid}", "left")
            self.t.insert("end-1c", "\n")
            self.t.mark_set("pstart", "end-1c")
            if int(self.t.index("end-1c").split(".")[0]) > 300:
                self.t.delete("1.0", "100.0")     # bound memory
                # trimmed seg marks collapse to line 1 but stay
                # registered — a late translation would land at the
                # top of the view
                for m in list(self.t.mark_names()):
                    if m.startswith("seg") and \
                            self.t.compare(m, "<=", "1.0"):
                        self.t.mark_unset(m)
        self._edit(go)

    def translation(self, sid, text):
        m = f"seg{sid}"
        if m in self.t.mark_names():
            self._edit(lambda: self.t.insert(m, "\n" + text, "tr"))

    def set_font(self, fs):
        self.t.configure(font=("Segoe UI", fs))
        self.t.tag_configure("tr", font=("Segoe UI", fs - 3))


class Overlay:
    def __init__(self, gui_q, status_items, mute, den_off,
                 mic_active=True):
        self.q = gui_q
        self.mute, self.den_off = mute, den_off
        self.mic_active = mic_active
        self.me = ""
        self.seg = 0
        self.finals = deque(maxlen=50)
        self._shown = {}                   # last text pushed per label
        self._mv_t = 0.0                   # drag/resize throttle
        self.fs, self.alpha = 11, 0.85
        _dpi_aware()
        r = self.root = tk.Tk()
        # fonts scale with DPI automatically; pixel constants don't —
        # scale them by the same factor (1.0 at 96 DPI)
        self.dpi = r.winfo_fpixels("1i") / 96
        r.overrideredirect(True)
        r.attributes("-topmost", True, "-alpha", self.alpha)
        r.configure(bg=BG)
        geo = self._load_geo()
        w, h = geo.get("w", int(560 * self.dpi)), \
            geo.get("h", int(240 * self.dpi))
        h = max(h, int(200 * self.dpi))  # saved geo may predate the
                                         # scrollback — never squeeze
                                         # it to a sliver
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
        self.view = CaptionView(r, lines=6, fs=self.fs)
        self.me_now = mk(fg=DIM, font=("Segoe UI", 10))
        self.text_lbls = (self.esc_lbl, self.me_now)
        self.esc_lbl.pack(fill="x", padx=12)
        self.view.t.pack(fill="both", expand=True)
        self.me_now.pack(fill="x", padx=12)
        grip = mk(fg=MUTE, font=("Segoe UI", 8), text="◢",
                  cursor="size_nw_se")
        self._grip = grip
        grip.pack(anchor="se", padx=4)
        grip.bind("<Button-1>", self._rs_down)
        grip.bind("<B1-Motion>", self._rs_move)

        for wdg in (r, top, self.badges, *self.text_lbls):
            wdg.bind("<Button-1>", self._down)
            wdg.bind("<B1-Motion>", self._move)
        r.bind("<Configure>", self._resized)
        r.bind("<Escape>", self._esc)
        r.bind("m", lambda e: self.tgl_mute())
        r.bind("d", lambda e: self.tgl_den())
        r.bind("c", lambda e: self.copy_last())
        r.bind("<MouseWheel>", self._wheel)
        # wheel over the text scrolls history (plain) or rescales
        # (ctrl) — must break, or the root handler also fades opacity
        self.view.t.bind("<MouseWheel>", self._text_wheel)
        signal.signal(signal.SIGINT, self._sigint)
        r.update_idletasks()
        r.report_callback_exception = lambda t, v, tb: diag_exc(
            "tk callback", t, v, tb)
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
            self._set_state("cap", "info", "hidden")
        else:
            # info, not warn — it isn't actionable, and an amber dot
            # all session would cry wolf
            self._set_state("cap", "info", "captured")
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
        """Manual drag — throttled to ~30Hz. The native HTCAPTION
        modal loop nested a Windows pump inside SendMessage, which
        re-entered Tcl and killed the process silently (console off)."""
        # toplevel bindtags deliver child-widget presses here too —
        # arming a drag on a Text/Button press would also select the
        # text AND move the window
        if isinstance(e.widget, tk.Button) or e.widget in (
                self._grip, self.view.t):
            self._dx = None              # this press must not drag
            return
        self._dx, self._dy = e.x_root - self.root.winfo_x(), \
            e.y_root - self.root.winfo_y()

    def _move(self, e):
        if getattr(self, "_dx", None) is None:
            return
        now = time.monotonic()
        if now - self._mv_t < 0.033:     # coalesce motion floods —
            return                       # geometry() is the GIL-cost
        self._mv_t = now
        self.root.geometry(
            f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _rs_down(self, e):
        self._rx, self._ry = e.x_root, e.y_root
        self._rw, self._rh = (self.root.winfo_width(),
                              self.root.winfo_height())

    def _rs_move(self, e):
        now = time.monotonic()
        if now - self._mv_t < 0.033:
            return
        self._mv_t = now
        w = max(int(360 * self.dpi), self._rw + e.x_root - self._rx)
        h = max(int(140 * self.dpi), self._rh + e.y_root - self._ry)
        self.root.geometry(f"{w}x{h}")

    def _resized(self, e):
        """Wrap labels to the new width and clamp a minimum size —
        native resize doesn't run our old _rs_move per-event code."""
        if e.widget is not self.root:
            return
        wpl = e.width - int(24 * self.dpi)
        for lbl in self.text_lbls:
            lbl.config(wraplength=wpl)
        w = max(int(360 * self.dpi), e.width)
        h = max(int(140 * self.dpi), e.height)
        if (w, h) != (e.width, e.height):
            self.root.geometry(f"{w}x{h}")

    def _text_wheel(self, e):
        if e.state & 0x4:                    # ctrl: font size
            self._wheel(e)
        else:                                # plain: scroll history
            self.view.t.yview_scroll(
                -3 if e.delta > 0 else 3, "units")
        return "break"                       # skip root's opacity

    def _wheel(self, e):
        if e.state & 0x4:                        # ctrl+wheel: font size
            self.fs = max(8, min(20, self.fs + (1 if e.delta > 0 else -1)))
            self.view.set_font(self.fs)
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
        self.root.clipboard_append(self.finals[-1]
                                   if self.finals else "")

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
                        self.view.partial(text)
                    else:
                        self.me = "you: " + text
                elif ev[0] == "final":
                    _, tag, text, *rest = ev
                    self.finals.append(
                        ("them" if tag == "spk" else "you") + ": " + text)
                    if tag == "spk":
                        self.seg = rest[0] if rest else 0
                        self.view.final(text, self.seg)
                        self.me = ""         # stale you: line clears
                    else:                    # when their turn ends
                        self.me = "you: " + text
                elif ev[0] == "trans":
                    if len(ev) >= 3:
                        # attaches to its own segment line — a stale
                        # seg finds no mark and is dropped
                        self.view.translation(ev[2], ev[1])
                elif ev[0] == "hotkey":
                    if ev[1] == "mute":
                        self.tgl_mute()
                    elif ev[1] == "den":
                        self.tgl_den()
                elif ev[0] == "state":
                    # badge/dot only — never overwrite the caption
                    self._set_state(ev[1], ev[2],
                                    ev[3] if len(ev) > 3 else "")
                elif ev[0] == "err":
                    self.view.partial(str(ev[1]))
                    self._set_state("worker", "err", str(ev[1])[:40])
        except queue.Empty:
            pass
        except Exception:                    # one bad event must not
            pass                             # kill the drain loop
        if dirty:
            zwsp = "\u200b"
            # me_now is a label; config() re-lays out even for
            # identical text — skip it when nothing changed
            txt = self.me or zwsp
            if self._shown.get(self.me_now) != txt:
                self._shown[self.me_now] = txt
                self.me_now.config(text=txt)

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

    if not acquire_single_instance():
        raise SystemExit("livecap is already running")
    try:
        import ctypes
        # 15.6ms default timer granularity ~= a whole audio frame —
        # tighten it for the queue timeouts and waits everywhere
        ctypes.windll.winmm.timeBeginPeriod(1)
        # audio path priority over everything else in the process
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass

    _dpi_aware()                       # before ANY window exists
    splash = loading_splash()        # models take seconds to load —
                                     # show something immediately

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

    def _load_models():
        ensure_asr_model()               # downloads if missing —
        model_dir = find_asr_dir()       # splash stays up via pump
        # preflight before any sherpa call — a missing file exits
        # silently from native code when the console is hidden
        missing = preflight(ROOT, model_dir)
        if missing:
            raise SystemExit("missing model files:\n  "
                             + "\n  ".join(str(m) for m in missing))
        den = make_denoiser()
        diag(f"ASR model: {model_dir.name}")
        rec, asr_provider = make_recognizer(model_dir)
        diag(f"ASR provider: {asr_provider}")
        rec2, _ = make_recognizer(model_dir)   # 2nd instance — memory
        return den, rec, rec2                  # is cheap, CPU isn't:
    try:                                     # decode can't serialize
        pump = (lambda: splash.update()) if splash else             (lambda: time.sleep(0.05))
        den, rec, rec2 = run_with_pump(_load_models, pump)
    except BaseException:
        drop_splash()                    # splash must not outlive us
        raise
    pipe = Pipeline(rec)                 # spk stream
    pipe_mic = Pipeline(rec2)            # mic stream — own lock pool

    status = []
    # queues first — the cable's finished_callback reports to gui_q
    mic_q = queue.Queue(maxsize=10)      # voice path: <160ms, drop not delay
    micasr_q = queue.Queue(maxsize=64)   # ~1s of 16ms chunks
    qs = queue.Queue(maxsize=64)         # ~2s of 32ms chunks
    gui_q = queue.Queue(maxsize=1024)     # bounded — a stalled overlay
                                          # can't eat RAM
    rep = Reporter(gui_q)                  # ONE dedupe memory shared
                                          # by engine, supervisor, cb
    ring, out_stream, cable_sr = None, None, SR
    # shared stream state — the supervisor swaps these on reopen and
    # the voice engine picks them up per chunk
    state = {"in": None, "out": None, "ring": None, "mic_sr": SR,
             "cable_sr": SR, "mic_dead": False, "cable_dead": False,
             "reinit_t": 0.0,
             "ev": threading.Event(), "cands": []}
    cands = cable_candidates()
    state["cands"] = cands
    if not cands:
        state["cable_absent"] = True
        status.append(("cable","warn","no VB-CABLE"))
        diag("VB-CABLE not found — captions only, no mic passthrough")
    elif not cable_usable():
        state["cable_absent"] = True
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
                finished_callback=dead_cb(
                    state, "cable", rep, "cable stream stopped"),
                **kw)
        ring, out_stream, cable_sr = open_cable(
            cands, True, _open_cable)
        if out_stream is None:
            status.append(("cable","err","cable open failed"))
            mark_dead(state, "cable", rep, "cable open failed")
        else:
            status.append(("cable","ok","cable ✓"))
            diag(f"cable latency: {out_stream.latency}")
    state.update({"out": out_stream, "ring": ring,
                  "cable_sr": cable_sr})

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
    trans_q = queue.Queue(maxsize=16) if args.lang else None

    in_stream, mic_sr = None, SR
    if out_stream is None:
        status.append(("mic","warn","mic off"))
    else:
        mic_idx = pick_mic()
        if mic_idx is None:
            status.append(("mic","warn","no mic"))
            mark_dead(state, "mic", rep, "no mic")
        else:
         try:
            st, mic_sr = open_mic(
                mic_idx, den, mic_q, gui_q,
                dead=dead_cb(state, "mic", rep,
                             "mic stream stopped"))
            st.start()
            in_stream = st                 # assign only after start()
            state["in"], state["mic_sr"] = st, mic_sr
            status.append(("mic","ok","mic ✓"))
            diag(f"mic open: dev{mic_idx} {mic_sr}Hz "
                 f"latency={st.latency}")
         except Exception as e:
            status.append(("mic","warn","no mic"))
            diag(f"no mic input ({e})")
            mark_dead(state, "mic", rep, "mic open failed")
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
            state["ring"] = state["out"] = None
            mark_dead(state, "cable", rep, "cable start failed")
            status.append(("cable","err","start failed"))
    workers = []
    worker_defs = []
    hk_tid = []                    # hotkey worker's thread id — the
                                   # WM_QUIT post at shutdown needs it
    # engine runs even when the mic/cable failed at startup — its
    # stall detector is what flags them dead and the supervisor is
    # what brings them back; the engine picks up new ring/rates from
    # `state` per chunk
    worker_defs.append(
        (voice_engine, "voice",
         dict(args=(mic_q, den, ring, micasr_q, mute, den_off,
                    gui_q),
              kwargs=dict(mic_sr=mic_sr, cable_sr=cable_sr,
                          state=state, rep=rep))))
    worker_defs += [
            (stream_supervisor, "stream-sup",
             dict(args=(state, state["ev"], gui_q, den, mic_q),
                  kwargs=dict(rep=rep))),
            (loopback_pump, "spk-pump", dict(args=(qs, gui_q))),
            (asr_consumer, "asr-spk",
             dict(args=(pipe, qs, gui_q, log, log_lock,
                        "spk", trans_q))),
            (asr_consumer, "asr-mic",
             dict(args=(pipe_mic, micasr_q, gui_q, log,
                        log_lock, "mic"))),
            (hotkey_worker, "hotkey",
             dict(args=(gui_q,), kwargs=dict(tid_out=hk_tid))),
    ]
    for tgt, name, kw in worker_defs:
        t = threading.Thread(target=tgt, daemon=True, name=name, **kw)
        t.start()
        workers.append(t)
    try:                                 # startup allocations done —
        import gc                        # a gen-2 sweep mid-call is a
        gc.freeze()                      # hiccup on the audio path
    except Exception:
        pass
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

    diag("livecap running — mic -> denoise -> cable, captions for both")
    try:
        drop_splash()
        # mic controls always visible — the supervisor may bring the
        # leg up later; a dead mic shows as badges, not hidden buttons
        Overlay(gui_q, status, mute, den_off, mic_active=True).run()
    finally:
        stop.set()
        try:
            import ctypes
            if hk_tid:                   # wake the hotkey thread's
                ctypes.windll.user32.PostThreadMessageW(  # GetMessage
                    hk_tid[0], 0x0012, 0, 0)   # WM_QUIT — it unregisters
        except Exception:                # + joins; daemon dies anyway
            pass
        for w in workers:                # let workers flush + emit
            w.join(timeout=3)
            if w.is_alive():
                diag(f"shutdown: {w.name} still blocked after 3s")
        try:
            log.close()
        except Exception:
            pass
        for st in (state["out"], state["in"]):  # live streams, not
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
    drop_splash()
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
