#!/usr/bin/env python
"""livecap — Krisp-style live captions + mic denoise passthrough.

  mic (16k) -> GTCRN denoise -> VB-CABLE virtual mic (what call apps hear)
                             -> tap -> [you] caption
  speaker loopback -> streaming ASR -> [them] caption

Needs VB-CABLE installed and selected as the mic in your call app for the
passthrough half; captions work either way. Esc or close the window to quit.
"""

import queue
import signal
import sys
import threading
import tkinter as tk
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


def voice_engine(mic_q, den, ring, cable_sr, micasr_q):
    """mic -> denoise -> ring buffer for cable + tap into ASR queue."""
    while not stop.is_set():
        try:
            x = mic_q.get(timeout=0.1)
        except queue.Empty:
            continue
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

def asr_consumer(pipe, qs, qm, gui_q):
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
            gui_q.put(("final", tag, final))
        elif pipe.partials.get(tag):
            gui_q.put(("partial", tag, pipe.partials[tag]))
    pipe.flush()


# ------------------------------------------------------------------ overlay

BG, DIM, TXT, MUTE = "#101418", "#9ca3af", "#f9fafb", "#4b5563"


class Overlay:
    def __init__(self, gui_q):
        self.q = gui_q
        self.them = {"last": "", "now": ""}
        self.me = ""
        r = self.root = tk.Tk()
        r.overrideredirect(True)
        r.attributes("-topmost", True, "-alpha", 0.85)
        r.configure(bg=BG)
        w, h = 920, 100
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - h - 90}")
        mk = lambda **kw: tk.Label(r, bg=BG, anchor="w", justify="left",
                                   wraplength=w - 24, **kw)
        self.them_last = mk(fg=MUTE, font=("Segoe UI", 10))
        self.them_now = mk(fg=TXT, font=("Segoe UI", 12, "bold"))
        self.me_now = mk(fg=DIM, font=("Segoe UI", 10))
        for lbl in (self.them_last, self.them_now, self.me_now):
            lbl.pack(fill="x", padx=12)
        mk(fg=MUTE, font=("Segoe UI", 8),
           text="drag to move · esc to quit").pack(anchor="se", padx=8)
        for wdg in (r, *r.winfo_children()):
            wdg.bind("<Button-1>", self._down)
            wdg.bind("<B1-Motion>", self._move)
        r.bind("<Escape>", lambda e: self.close())
        signal.signal(signal.SIGINT, self._sigint)
        r.update_idletasks()
        # WDA_EXCLUDEFROMCAPTURE: visible on screen, invisible to
        # screenshots and screen-share capture (Windows 10 1903+)
        try:
            import ctypes
            ctypes.windll.user32.SetWindowDisplayAffinity(
                int(r.wm_frame(), 16), 0x11)
        except Exception:
            pass
        r.after(30, self._poll)

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
                    if tag == "spk":
                        self.them["last"], self.them["now"] = text, ""
                    else:
                        self.me = "you: " + text
                elif ev[0] == "err":
                    self.them["now"] = ev[1]
        except queue.Empty:
            pass
        if dirty:
            zwsp = "\u200b"
            self.them_last.config(text=self.them["last"] or zwsp)
            self.them_now.config(text=self.them["now"] or zwsp)
            self.me_now.config(text=self.me or zwsp)
        if not stop.is_set():
            self.root.after(30, self._poll)

    def close(self):
        stop.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# --------------------------------------------------------------------- main

def main():
    den = make_denoiser()
    model_dir = asr_dir()
    print(f"ASR model: {model_dir.name}", file=sys.stderr)
    rec = make_recognizer(model_dir)
    punct = None if "nemotron" in model_dir.name.lower() else make_punctuator()
    pipe = Pipeline(rec, punct)

    cable_idx, cable_sr = find_cable()
    ring = out_stream = None
    if cable_idx is None:
        print("VB-CABLE not found — captions only, no mic passthrough",
              file=sys.stderr)
    else:
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

    in_stream = None
    try:
        in_stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32",
            blocksize=den.frame_shift_in_samples,
            callback=lambda d, f, t, s: put_drop(mic_q, d[:, 0].copy()))
        in_stream.start()
    except Exception as e:
        print(f"no mic input ({e}) — speaker captions only", file=sys.stderr)
    if out_stream is not None:
        out_stream.start()
    threading.Thread(target=voice_engine, daemon=True,
                     args=(mic_q, den, ring, cable_sr, micasr_q)).start()
    threading.Thread(target=loopback_pump, daemon=True, args=(qs,)).start()
    threading.Thread(target=asr_consumer, daemon=True,
                     args=(pipe, qs, micasr_q, gui_q)).start()

    print("livecap running — mic -> denoise -> cable, captions for both",
          file=sys.stderr)
    try:
        Overlay(gui_q).run()
    finally:
        stop.set()
        if out_stream is not None:
            out_stream.stop()
            out_stream.close()
        if in_stream is not None:
            in_stream.stop()
            in_stream.close()


if __name__ == "__main__":
    main()
