#!/usr/bin/env python
"""Real-time mic -> speech denoise -> streaming ASR -> punctuation.

CPU-only pipeline on Windows:
  mic (16 kHz) -> [GTCRN | DPDFNet denoiser] -> streaming Zipformer transducer
               -> online punctuation/case restoration -> stdout

Usage:
  python transcribe.py                      # mic -> asr -> punct
  python transcribe.py --denoiser dpdfnet   # heavier, better denoise
  python transcribe.py --denoiser none      # no denoise
  python transcribe.py --file some.wav      # process a wav file instead of mic
  python transcribe.py --save-clean out.wav # dump denoised 16 kHz audio
  python transcribe.py --loopback           # transcribe speaker output, not mic
  python transcribe.py --both               # mic + speaker, tagged [mic]/[spk]
  python transcribe.py --model <dir-name>   # pick ASR model dir under models/
"""

import argparse
import os
import queue
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

ROOT = Path(__file__).resolve().parent
PUNCT_DIR = ROOT / "models" / "sherpa-onnx-online-punct-en-2024-08-06"
SR = 16000


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n_out = int(round(len(x) * dst_sr / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, len(x) - 1, n_out)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def make_denoiser(kind: str):
    if kind == "none":
        return None
    cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
    if kind == "gtcrn":
        cfg.model.gtcrn.model = str(ROOT / "models" / "gtcrn_simple.onnx")
    elif kind == "dpdfnet":
        cfg.model.dpdfnet.model = str(ROOT / "models" / "dpdfnet_baseline.onnx")
    else:
        raise ValueError(kind)
    cfg.model.num_threads = 1
    return sherpa_onnx.OnlineSpeechDenoiser(cfg)


def asr_dir(name: str | None) -> Path:
    if name:
        d = ROOT / "models" / name
        if not (d / "tokens.txt").is_file():
            raise SystemExit(f"model not found: {d}")
        return d
    models = [d for d in (ROOT / "models").iterdir()
              if (d / "tokens.txt").is_file()]
    # zipformer is the only bundled model that sustains real-time decode on a
    # modest cpu (nemotron-0.6b measures RTF ~2.5 on 4 cores)
    models.sort(key=lambda d: ("zipformer" not in d.name, d.name))
    if not models:
        raise SystemExit("no ASR model (with tokens.txt) under models/")
    return models[0]


def make_recognizer(model_dir: Path, threads: int):
    def pick(prefix):
        for pat in (f"{prefix}*.int8.onnx", f"{prefix}*.onnx"):
            fs = sorted(model_dir.glob(pat))
            if fs:
                return str(fs[0])
        raise SystemExit(f"no {prefix}*.onnx in {model_dir}")

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=pick("encoder"),
        decoder=pick("decoder"),
        joiner=pick("joiner"),
        num_threads=threads,
        sample_rate=SR,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
        decoding_method="greedy_search",
        provider="cpu",
    )


def make_punctuator():
    cfg = sherpa_onnx.OnlinePunctuationConfig()
    cfg.model_config.cnn_bilstm = str(PUNCT_DIR / "model.int8.onnx")
    cfg.model_config.bpe_vocab = str(PUNCT_DIR / "bpe.vocab")
    cfg.model_config.num_threads = 1
    return sherpa_onnx.OnlinePunctuation(cfg)


def result_text(res) -> str:
    return res if isinstance(res, str) else getattr(res, "text", str(res))


class WavWriter:
    def __init__(self, path: str):
        self.f = wave.open(path, "wb")
        self.f.setnchannels(1)
        self.f.setsampwidth(2)
        self.f.setframerate(SR)

    def write(self, samples: np.ndarray):
        pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
        self.f.writeframes(pcm.tobytes())

    def close(self):
        self.f.close()


class Pipeline:
    def __init__(self, denoiser, recognizer, punct, wav_out=None,
                 den_tag="mic"):
        self.den = denoiser      # applied to den_tag-tagged audio only
        self.den_tag = den_tag   # also the source recorded by --save-clean
        self.rec = recognizer
        self.punct = punct
        self.streams = {}   # tag -> recognizer stream
        self.partials = {}  # tag -> last partial text
        self.wav = WavWriter(wav_out) if wav_out else None

    def feed(self, samples: np.ndarray, tag: str = "mic"):
        """Feed 16 kHz float32 mono samples; returns list of finalized segments."""
        stream = self.streams.get(tag)
        if stream is None:
            stream = self.streams[tag] = self.rec.create_stream()
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if tag == self.den_tag and self.den is not None:
            out = self.den.run(samples, SR)
            samples = np.asarray(out.samples, dtype=np.float32)
        if self.wav is not None and tag == self.den_tag and len(samples):
            self.wav.write(samples)
        if not len(samples):
            return []
        stream.accept_waveform(SR, samples)
        while self.rec.is_ready(stream):
            self.rec.decode_stream(stream)
        finals = []
        if self.rec.is_endpoint(stream):
            text = result_text(self.rec.get_result(stream)).strip()
            self.rec.reset(stream)
            if text:
                finals.append(self.punct.add_punctuation_with_case(text.lower())
                              if self.punct else text)
            self.partials.pop(tag, None)
        else:
            self.partials[tag] = result_text(self.rec.get_result(stream)).strip()
        return finals

    def flush(self):
        """Finish all streams; returns list of (tag, text)."""
        if self.den is not None:
            out = self.den.flush()
            samples = np.asarray(out.samples, dtype=np.float32)
            tag = self.den_tag if self.den_tag in self.streams else \
                next(iter(self.streams), None)
            if len(samples) and tag:
                self.streams[tag].accept_waveform(SR, samples)
                if self.wav is not None and tag == self.den_tag:
                    self.wav.write(samples)
        # streaming transducers need right-context silence to emit their
        # final tokens; without it the last syllables are dropped
        pad = np.zeros(int(0.75 * SR), dtype=np.float32)
        texts = []
        for tag, stream in self.streams.items():
            stream.accept_waveform(SR, pad)
            stream.input_finished()
            while self.rec.is_ready(stream):
                self.rec.decode_stream(stream)
            t = result_text(self.rec.get_result(stream)).strip()
            if t:
                texts.append((tag, self.punct.add_punctuation_with_case(t.lower())
                              if self.punct else t))
        if self.wav:
            self.wav.close()
        return texts


def run_file(path: str, pipe: Pipeline):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
        sw = w.getsampwidth()
        if sw == 1:
            audio = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        elif sw == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            audio = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            audio = np.where(audio >= (1 << 23), audio - (1 << 24), audio)
            audio = audio.astype(np.float32) / 8388608.0
        elif sw == 4:
            audio = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width: {sw} bytes")
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
    audio = resample_linear(audio, sr, SR)
    shift = pipe.den.frame_shift_in_samples if pipe.den else 512
    try:
        for i in range(0, len(audio), shift):
            for seg in pipe.feed(audio[i:i + shift]):
                print(seg)
    finally:
        for _, t in pipe.flush():
            print(t)


def run_live(device, pipe: Pipeline, mode: str):
    """mode: 'mic' | 'spk' (speaker loopback) | 'both'."""
    shift = pipe.den.frame_shift_in_samples if pipe.den else 512
    # separate bounded queues: speaker audio is always drained before mic,
    # so under a backlog the (lower-priority) dictation drops first
    qs = queue.Queue(maxsize=512)
    qm = queue.Queue(maxsize=512)
    stats = {"dropped": 0, "mic_overruns": 0}
    tagged = mode == "both"

    def put(q, item):
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
            stats["dropped"] += 1

    def next_item():
        for q in (qs, qm):          # spk queue first, then mic
            try:
                return q.get_nowait()
            except queue.Empty:
                pass
        while True:                  # idle: block on spk, poll mic
            try:
                return qs.get(timeout=0.1)
            except queue.Empty:
                pass
            try:
                return qm.get_nowait()
            except queue.Empty:
                pass

    def cb(indata, frames, time_info, status):
        if status:
            stats["mic_overruns"] += 1
        put(qm, ("mic", indata[:, 0].copy()))

    def pump():
        import ctypes
        import warnings
        com_init = False
        try:
            import soundcard as sc
            ctypes.windll.ole32.CoInitialize(None)
            com_init = True
            warnings.filterwarnings("ignore", "data discontinuity")
            # a real mic can share the speaker's name (e.g. "Remote Audio"
            # on RDP) — get_microphone() then resolves to the mic, not the
            # loopback; match the speaker's endpoint id instead
            speaker = sc.default_speaker()
            mic = next((m for m in sc.all_microphones(include_loopback=True)
                        if m.isloopback and m.id == speaker.id), None)
            if mic is None:
                raise RuntimeError(
                    f"no loopback capture device for {speaker.name!r}")
            with mic.recorder(samplerate=SR, channels=1) as rec:
                while True:
                    put(qs, ("spk", rec.record(shift)[:, 0]))
        except Exception as e:
            put(qs, ("err", e))
        finally:
            if com_init:
                ctypes.windll.ole32.CoUninitialize()

    stream = None
    if mode in ("mic", "both"):
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                blocksize=shift, device=device, callback=cb)
    if mode in ("spk", "both"):
        threading.Thread(target=pump, daemon=True).start()

    name = sd.query_devices(device)["name"] if device is not None else "default"
    label = {"mic": name, "spk": "speaker loopback",
             "both": f"{name} + speaker"}[mode]
    den_state = f"on ({pipe.den_tag})" if pipe.den else "off"
    print(f"Listening ({label}, denoiser={den_state}). Ctrl+C to stop.")

    try:
        if stream is not None:
            stream.start()
        while True:
            tag, chunk = next_item()
            if tag == "err":
                sys.stdout.write(f"\x1b[2K\r[speaker capture failed: {chunk}]\n")
                break
            if stats["dropped"]:
                sys.stdout.write("\x1b[2K\r[asr behind: dropped "
                                 f"{stats['dropped']} chunks]\n")
                stats["dropped"] = 0
            if stats["mic_overruns"]:
                sys.stdout.write("\x1b[2K\r[mic overflowed "
                                 f"{stats['mic_overruns']}x]\n")
                stats["mic_overruns"] = 0
            finals = pipe.feed(chunk, tag)
            if finals:
                for seg in finals:
                    pre = f"[{tag}] " if tagged else ""
                    sys.stdout.write("\x1b[2K\r" + pre + seg + "\n")
            elif pipe.partials.get(tag):
                pre = f"{tag}> " if tagged else ""
                sys.stdout.write("\x1b[2K\r" + pre + pipe.partials[tag][:90])
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            if stream.active:
                stream.stop()
            stream.close()
        for tag, t in pipe.flush():
            pre = f"[{tag}] " if tagged else ""
            print("\n" + pre + t)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--denoiser", choices=["none", "gtcrn", "dpdfnet"],
                    default=None, help="denoiser for the captured source "
                                       "(default: none)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--loopback", action="store_true",
                    help="capture speaker output (WASAPI loopback) instead of mic")
    ap.add_argument("--both", action="store_true",
                    help="capture mic AND speaker output simultaneously")
    ap.add_argument("--model", help="ASR model dir name under models/ "
                                    "(default: prefer zipformer if present)")
    ap.add_argument("--file", help="process this wav instead of the mic")
    ap.add_argument("--save-clean", help="write denoised 16 kHz wav here")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if os.name == "nt":
        os.system("")  # enable ANSI escapes on conhost

    mode = "both" if args.both else "spk" if args.loopback else "mic"
    denoiser = args.denoiser or "none"

    den = make_denoiser(denoiser)
    model_dir = asr_dir(args.model)
    print(f"ASR model: {model_dir.name}", file=sys.stderr)
    rec = make_recognizer(model_dir, args.threads)
    # nemotron emits cased, punctuated text natively
    punct = None if "nemotron" in model_dir.name.lower() else make_punctuator()
    # denoiser (and --save-clean) target the mic feed, or the speaker feed
    # in speaker-only mode
    den_tag = "spk" if (mode == "spk" and not args.file) else "mic"
    pipe = Pipeline(den, rec, punct, args.save_clean, den_tag=den_tag)

    if args.file:
        run_file(args.file, pipe)
    else:
        run_live(args.device, pipe, mode)


if __name__ == "__main__":
    main()
