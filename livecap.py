#!/usr/bin/env python
"""livecap — Krisp-style live captions + mic denoise passthrough.

  mic (16k) -> GTCRN denoise -> VB-CABLE virtual mic (what call apps hear)
                             -> tap -> [you] caption
  speaker loopback -> streaming ASR -> [them] caption

Needs VB-CABLE installed and selected as the mic in your call app for the
passthrough half; captions work either way. Esc or close the window to quit.
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
from collections import deque
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

ROOT = Path(__file__).resolve().parent
PUNCT_DIR = ROOT / "models" / "sherpa-onnx-online-punct-en-2024-08-06"
SR = 16000

stop = threading.Event()


# ---------------------------------------------------------------- models

def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n_out = int(round(len(x) * dst_sr / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, len(x) - 1, n_out)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def asr_dir() -> Path:
    models = [d for d in (ROOT / "models").iterdir()
              if (d / "tokens.txt").is_file()]
    # zipformer is the only bundled model that sustains real-time on this cpu
    models.sort(key=lambda d: ("zipformer" not in d.name, d.name))
    if not models:
        raise SystemExit("no ASR model (with tokens.txt) under models/")
    return models[0]


def make_recognizer(model_dir: Path):
    def pick(prefix):
        for pat in (f"{prefix}*.int8.onnx", f"{prefix}*.onnx"):
            fs = sorted(model_dir.glob(pat))
            if fs:
                return str(fs[0])
        raise SystemExit(f"no {prefix}*.onnx in {model_dir}")

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
        num_threads=4, sample_rate=SR, feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
        decoding_method="greedy_search", provider="cpu",
    )


def make_denoiser():
    cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
    cfg.model.gtcrn.model = str(ROOT / "models" / "gtcrn_simple.onnx")
    cfg.model.num_threads = 1
    return sherpa_onnx.OnlineSpeechDenoiser(cfg)


def make_punctuator():
    cfg = sherpa_onnx.OnlinePunctuationConfig()
    cfg.model_config.cnn_bilstm = str(PUNCT_DIR / "model.int8.onnx")
    cfg.model_config.bpe_vocab = str(PUNCT_DIR / "bpe.vocab")
    cfg.model_config.num_threads = 1
    return sherpa_onnx.OnlinePunctuation(cfg)


def result_text(res) -> str:
    return res if isinstance(res, str) else getattr(res, "text", str(res))


def put_drop(q: queue.Queue, item):
    """Non-blocking put; on full, drop the oldest chunk to bound latency."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


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
    return code if "-" in code else f"{code}-{code.upper()}"


def translator_worker(tq, gui_q, log, log_lock, lang, email):
    """Translate finalized [them] segments off the caption critical path."""
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    try:
        tr = MyMemoryTranslator(source="en-US", target=lang,
                                email=email)
        fallback = GoogleTranslator(source="en", target=lang.split("-")[0])
    except Exception as e:
        gui_q.put(("trans", f"[translation unavailable: {e}]"))
        return
    warned = False
    while not stop.is_set():
        try:
            text = tq.get(timeout=0.1)
        except queue.Empty:
            continue
        while True:                        # keep only the newest segment
            try:
                text = tq.get_nowait()
            except queue.Empty:
                break
        try:
            out = tr.translate(text)
        except Exception:
            try:
                out = fallback.translate(text)
            except Exception:
                out = None
        if out:
            warned = False
            gui_q.put(("trans", out))
            with log_lock:
                log.write(f"    -> {out}\n")
        elif not warned:                   # visible once, not per call
            warned = True
            gui_q.put(("trans", "[translation paused — rate limit]"))


# ------------------------------------------------------------------- asr

class Pipeline:
    def __init__(self, recognizer, punct):
        self.rec = recognizer
        self.punct = punct
        self.streams = {}
        self.partials = {}

    def feed(self, samples: np.ndarray, tag: str):
        stream = self.streams.get(tag)
        if stream is None:
            stream = self.streams[tag] = self.rec.create_stream()
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if not len(samples):
            return
        stream.accept_waveform(SR, samples)
        while self.rec.is_ready(stream):
            self.rec.decode_stream(stream)
        if self.rec.is_endpoint(stream):
            text = result_text(self.rec.get_result(stream)).strip()
            self.rec.reset(stream)
            self.partials.pop(tag, None)
            if text:
                return (self.punct.add_punctuation_with_case(text.lower())
                        if self.punct else text)
        else:
            self.partials[tag] = result_text(
                self.rec.get_result(stream)).strip()

    def flush(self):
        pad = np.zeros(int(0.75 * SR), dtype=np.float32)
        for stream in self.streams.values():
            stream.accept_waveform(SR, pad)
            stream.input_finished()
            while self.rec.is_ready(stream):
                self.rec.decode_stream(stream)


# ------------------------------------------------------ mic -> denoise ->
#                                                cable + [you] caption tap

class Ring:
    """Float32 ring buffer feeding the cable output callback."""

    def __init__(self, n: int):
        self.b = np.zeros(n, np.float32)
        self.n = n
        self.r = self.w = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        n = len(x)
        if n > self.n:                    # keep only the newest samples
            x = x[n - self.n:]
            n = self.n
        with self.lock:
            if n > self.n - (self.w - self.r):   # overflow: drop oldest
                self.r += n - (self.n - (self.w - self.r))
            i = self.w % self.n
            k = min(n, self.n - i)
            self.b[i:i + k] = x[:k]
            self.b[:n - k] = x[k:]
            self.w += n

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self.lock:
            have = min(n, self.w - self.r)
            i = self.r % self.n
            k = min(have, self.n - i)
            out[:k] = self.b[i:i + k]
            out[k:have] = self.b[:have - k]
            self.r += have
        return out


def find_cable():
    """VB-CABLE playback endpoint 'CABLE Input' (calls read 'CABLE Output')."""
    for i, d in enumerate(sd.query_devices()):
        if "cable input" in d["name"].lower() and d["max_output_channels"] > 0:
            return i, int(d["default_samplerate"])
    return None, None


def voice_engine(mic_q, den, ring, cable_sr, micasr_q, mute, den_off):
    """mic -> denoise -> ring buffer for cable + tap into ASR queue."""
    needs_reset = False
    while not stop.is_set():
        try:
            x = mic_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if mute.is_set():                    # hard mute: nothing leaves
            needs_reset = True
            continue
        if den_off.is_set():                 # passthrough without denoise
            y = np.ascontiguousarray(x, np.float32)
            needs_reset = True
        else:
            if needs_reset:                  # resync denoiser frame state
                den.reset()
                needs_reset = False
            out = den.run(np.ascontiguousarray(x, np.float32), SR)
            y = np.asarray(out.samples, dtype=np.float32)
        if not len(y):
            continue
        if ring is not None:
            ring.push(resample_linear(y, SR, cable_sr))
        put_drop(micasr_q, y)


# -------------------------------------------------------- speaker -> [them]

def loopback_pump(qs):
    import ctypes
    com_init = False
    try:
        import warnings
        import soundcard as sc
        ctypes.windll.ole32.CoInitialize(None)
        com_init = True
        warnings.filterwarnings("ignore", "data discontinuity")
        # real mics can share the speaker's name ("Remote Audio" on RDP) —
        # match the loopback by the speaker's endpoint id, not by name
        speaker = sc.default_speaker()
        mic = next((m for m in sc.all_microphones(include_loopback=True)
                    if m.isloopback and m.id == speaker.id), None)
        if mic is None:
            raise RuntimeError(
                f"no loopback capture device for {speaker.name!r}")
        with mic.recorder(samplerate=SR, channels=1) as rec:
            while not stop.is_set():
                put_drop(qs, rec.record(512)[:, 0])
    except Exception as e:
        put_drop(qs, ("err", e))
    finally:
        if com_init:
            ctypes.windll.ole32.CoUninitialize()


# ------------------------------------------------------------- asr consumer

def asr_consumer(pipe, qs, qm, gui_q, log, log_lock, trans_q):
    def get():
        for tag, q in (("spk", qs), ("mic", qm)):   # speaker first, always
            try:
                return tag, q.get_nowait()
            except queue.Empty:
                pass
        while not stop.is_set():
            try:
                return "spk", qs.get(timeout=0.1)
            except queue.Empty:
                pass
            try:
                return "mic", qm.get_nowait()
            except queue.Empty:
                pass
        return None

    while not stop.is_set():
        item = get()
        if item is None:
            break
        tag, chunk = item
        if isinstance(chunk, tuple):      # ("err", exc) sentinel from pump
            gui_q.put(("err", f"speaker capture failed: {chunk[1]}"))
            continue
        final = pipe.feed(chunk, tag)
        if final:
            with log_lock:
                log.write(f"[{time.strftime('%H:%M:%S')}] "
                          f"[{'them' if tag == 'spk' else 'you'}] {final}\n")
            if trans_q is not None and tag == "spk":
                put_drop(trans_q, final)
            gui_q.put(("final", tag, final))
        elif pipe.partials.get(tag):
            p = pipe.partials[tag]
            # raw zipformer partials are ALL-CAPS — sentence-case them for
            # display (leave already-cased models like nemotron alone)
            if p.isupper():
                p = p.capitalize()
            gui_q.put(("partial", tag, p))
    pipe.flush()


# ------------------------------------------------------------------ overlay

BG, DIM, TXT, MUTE, ACC = "#101418", "#9ca3af", "#f9fafb", "#4b5563", "#93c5fd"
LOGS = ROOT / "logs"
CFG = LOGS / "livecap-window.json"


class Overlay:
    def __init__(self, gui_q, status, mute, den_off):
        self.q = gui_q
        self.mute, self.den_off = mute, den_off
        self.them = {"last": "", "now": ""}
        self.me = ""
        self.tr = ""
        self.finals = deque(maxlen=50)
        self.fs, self.alpha = 11, 0.85
        r = self.root = tk.Tk()
        r.overrideredirect(True)
        r.attributes("-topmost", True, "-alpha", self.alpha)
        r.configure(bg=BG)
        geo = self._load_geo()
        w, h = geo.get("w", 920), geo.get("h", 100)
        x = geo.get("x", (r.winfo_screenwidth() - w) // 2)
        y = geo.get("y", r.winfo_screenheight() - h - 90)
        r.geometry(f"{w}x{h}+{x}+{y}")
        mk = lambda **kw: tk.Label(r, bg=BG, anchor="w", justify="left",
                                   wraplength=w - 24, **kw)

        top = tk.Frame(r, bg=BG)
        top.pack(fill="x")
        self.status = tk.Label(top, text=status, fg=ACC, bg=BG,
                               anchor="w", font=("Segoe UI", 9))
        self.status.pack(side="left", padx=12)
        self.mute_lbl = self._ctl(top, "mic: on", self.tgl_mute)
        self.den_lbl = self._ctl(top, "denoise: on", self.tgl_den)
        self._ctl(top, "copy all", self.copy_all)
        self._ctl(top, "copy", self.copy_last)
        self._ctl(top, "×", self.close)

        self.them_last = mk(fg=DIM, font=("Segoe UI", 10))
        self.them_now = mk(fg=TXT, font=("Segoe UI", 11))
        self.trans = mk(fg=ACC, font=("Segoe UI", 10))
        self.me_now = mk(fg=DIM, font=("Segoe UI", 10))
        self.text_lbls = (self.them_last, self.them_now, self.trans,
                          self.me_now)
        for lbl in self.text_lbls:
            lbl.pack(fill="x", padx=12)
        grip = mk(fg=MUTE, font=("Segoe UI", 8), text="◢",
                  cursor="size_nw_se")
        grip.pack(anchor="se", padx=4)
        grip.bind("<Button-1>", self._rs_down)
        grip.bind("<B1-Motion>", self._rs_move)

        for wdg in (r, top, self.status, *self.text_lbls):
            wdg.bind("<Button-1>", self._down)
            wdg.bind("<B1-Motion>", self._move)
        r.bind("<Escape>", lambda e: self.close())
        r.bind("m", lambda e: self.tgl_mute())
        r.bind("d", lambda e: self.tgl_den())
        r.bind("c", lambda e: self.copy_last())
        r.bind("<MouseWheel>", self._wheel)
        signal.signal(signal.SIGINT, self._sigint)
        r.update_idletasks()
        # WDA_EXCLUDEFROMCAPTURE: on screen normally, invisible to
        # screenshots and screen-share capture (Windows 10 1903+)
        hidden = False
        try:
            import ctypes
            hidden = bool(ctypes.windll.user32.SetWindowDisplayAffinity(
                int(r.wm_frame(), 16), 0x11))
        except Exception:
            pass
        if hidden:
            self.status.config(text=status + " · hidden-from-capture ✓")
        r.after(30, self._poll)
        r.after(9000, lambda: self.status.config(text=""))

    def _ctl(self, parent, text, fn):
        lbl = tk.Label(parent, text=text, fg=DIM, bg=BG,
                       font=("Segoe UI", 9), cursor="hand2")
        lbl.pack(side="right", padx=5, pady=2)
        lbl.bind("<Button-1>", lambda e: fn())
        return lbl

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
        w = max(360, self._rw + e.x_root - self._rx)
        h = max(60, self._rh + e.y_root - self._ry)
        self.root.geometry(f"{w}x{h}")
        for lbl in self.text_lbls:
            lbl.config(wraplength=w - 24)

    def _wheel(self, e):
        if e.state & 0x4:                        # ctrl+wheel: font size
            self.fs = max(8, min(20, self.fs + (1 if e.delta > 0 else -1)))
            self.them_last.config(font=("Segoe UI", self.fs - 1))
            self.them_now.config(font=("Segoe UI", self.fs))
            self.me_now.config(font=("Segoe UI", self.fs - 1))
        else:                                    # wheel: opacity
            self.alpha = max(0.35, min(1.0,
                                       self.alpha + (0.05 if e.delta > 0
                                                     else -0.05)))
            self.root.attributes("-alpha", self.alpha)

    def tgl_mute(self):
        on = self.mute.is_set()                  # muted now? -> unmute
        (self.mute.clear if on else self.mute.set)()
        self.mute_lbl.config(text="mic: on" if on else "mic: muted")

    def tgl_den(self):
        on = self.den_off.is_set()               # denoise off? -> on
        (self.den_off.clear if on else self.den_off.set)()
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
                elif ev[0] == "err":
                    self.them["now"] = ev[1]
        except queue.Empty:
            pass
        if dirty:
            zwsp = "\u200b"
            self.them_last.config(text=self.them["last"] or zwsp)
            self.them_now.config(text=self.them["now"] or zwsp)
            self.trans.config(text=self.tr or zwsp)
            self.me_now.config(text=self.me or zwsp)
            # grow the window to fit wrapped text (never auto-shrinks)
            self.root.update_idletasks()
            rh = self.root.winfo_reqheight()
            if rh > self.root.winfo_height() + 2:
                self.root.geometry(f"{self.root.winfo_width()}x{rh}"
                                   f"+{self.root.winfo_x()}"
                                   f"+{self.root.winfo_y()}")
        if not stop.is_set():
            self.root.after(30, self._poll)

    def close(self):
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
    ap.add_argument("--trans-email", metavar="EMAIL",
                    default=os.environ.get("LIVECAP_TRANS_EMAIL"),
                    help="email for MyMemory — raises free quota ~10x "
                         "(or set LIVECAP_TRANS_EMAIL)")
    args = ap.parse_args()

    den = make_denoiser()
    model_dir = asr_dir()
    print(f"ASR model: {model_dir.name}", file=sys.stderr)
    rec = make_recognizer(model_dir)
    punct = None if "nemotron" in model_dir.name.lower() else make_punctuator()
    pipe = Pipeline(rec, punct)

    status = []
    cable_idx, cable_sr = find_cable()
    ring = out_stream = None
    if cable_idx is None:
        status.append("no VB-CABLE — install it, then pick 'CABLE "
                      "Output' as your mic in the call app")
        print("VB-CABLE not found — captions only, no mic passthrough",
              file=sys.stderr)
    else:
        status.append("cable ✓")
        ring = Ring(cable_sr // 2)  # 0.5s buffer
        out_stream = sd.OutputStream(
            device=cable_idx, samplerate=cable_sr, channels=1,
            dtype="float32",
            callback=lambda data, frames, t, s:
                data.__setitem__((slice(None), 0), ring.pull(frames)))

    mic_q = queue.Queue(maxsize=512)
    micasr_q = queue.Queue(maxsize=512)   # post-denoise, lowest priority
    qs = queue.Queue(maxsize=512)         # speaker, highest priority
    gui_q = queue.Queue()
    mute, den_off = threading.Event(), threading.Event()
    log_lock = threading.Lock()
    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / f"livecap-{time.strftime('%Y%m%d-%H%M%S')}.txt",
               "w", encoding="utf-8", buffering=1)
    trans_q = queue.Queue(maxsize=4) if args.lang else None

    in_stream = None
    try:
        in_stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32",
            blocksize=den.frame_shift_in_samples,
            callback=lambda d, f, t, s: put_drop(mic_q, d[:, 0].copy()))
        in_stream.start()
        status.append("mic ✓")
    except Exception as e:
        status.append("no mic — speaker captions only")
        print(f"no mic input ({e})", file=sys.stderr)
    if out_stream is not None:
        out_stream.start()
    threading.Thread(target=voice_engine, daemon=True,
                     args=(mic_q, den, ring, cable_sr, micasr_q,
                           mute, den_off)).start()
    threading.Thread(target=loopback_pump, daemon=True, args=(qs,)).start()
    threading.Thread(target=asr_consumer, daemon=True,
                     args=(pipe, qs, micasr_q, gui_q, log, log_lock,
                           trans_q)).start()
    if trans_q is not None:
        status.append(f"trans→{args.lang}")
        threading.Thread(target=translator_worker, daemon=True,
                         args=(trans_q, gui_q, log, log_lock,
                               lang_code(args.lang),
                               args.trans_email)).start()

    status.append(model_dir.name[:30])
    print("livecap running — mic -> denoise -> cable, captions for both",
          file=sys.stderr)
    try:
        Overlay(gui_q, " · ".join(status), mute, den_off).run()
    finally:
        stop.set()
        log.close()
        if out_stream is not None:
            out_stream.stop()
            out_stream.close()
        if in_stream is not None:
            in_stream.stop()
            in_stream.close()


if __name__ == "__main__":
    main()
