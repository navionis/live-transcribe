#!/usr/bin/env python
"""Unit tests for livecap — pure logic only, no audio devices needed.

Run: .venv\\Scripts\\python.exe -m unittest test_livecap -v
"""

import ctypes
import io
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import livecap


class ResampleDownTest(unittest.TestCase):
    """N1 — downsampling must never read a future sample. On a ramp,
    linear interpolation is exact, so output diffs must equal step
    within fp error; the floor-based n_out produced a backward jump
    at every chunk boundary."""

    def _ramp_diffs(self, src, dst, chunk, nchunks):
        rs = livecap.Resampler(src, dst)
        x = np.arange(chunk * nchunks, dtype=np.float32)
        outs = [rs.resample(x[i:i + chunk])
                for i in range(0, len(x), chunk)]
        y = np.concatenate(outs)
        # d[1:]: the very first diff carries the initialized tail
        # (sample -1 is unknown) — a one-time artifact, not the
        # per-chunk-boundary glitch N1 produced
        return np.diff(y)[1:], rs.step, y

    def test_44k1_to_16k(self):
        d, step, _ = self._ramp_diffs(44100, 16000, 705, 30)
        self.assertTrue(np.allclose(d, step, rtol=0.01),
                        f"max dev {np.abs(d - step).max()}")

    def test_22k05_to_16k(self):
        d, step, _ = self._ramp_diffs(22050, 16000, 706, 30)
        self.assertTrue(np.allclose(d, step, rtol=0.01),
                        f"max dev {np.abs(d - step).max()}")

    def test_48k_to_16k(self):
        d, step, _ = self._ramp_diffs(48000, 16000, 705, 30)
        self.assertTrue(np.allclose(d, step, rtol=0.01),
                        f"max dev {np.abs(d - step).max()}")

    def test_total_output_count(self):
        _, step, y = self._ramp_diffs(44100, 16000, 705, 100)
        self.assertLess(abs(len(y) - 705 * 100 / step), 3)


class SteerTest(unittest.TestCase):
    """B3 — the controller must slow production when the ring is
    overfull (larger step) and speed it up when underfull."""

    def test_overfull_increases_step(self):
        rs = livecap.Resampler(16000, 44100)
        base = rs.step
        rs.steer(10000, 44100)              # ring overfull
        self.assertGreater(rs.step, base)

    def test_underfull_decreases_step(self):
        rs = livecap.Resampler(16000, 44100)
        base = rs.step
        rs.steer(-10000, 44100)             # ring starving
        self.assertLess(rs.step, base)

    def test_deadband(self):
        rs = livecap.Resampler(16000, 44100)
        base = rs.step
        rs.steer(10, 44100)                 # << 5ms deadband
        self.assertEqual(rs.step, base)

    def test_cap(self):
        rs = livecap.Resampler(16000, 44100)
        rs.steer(10 ** 9, 44100)
        self.assertAlmostEqual(rs.step, rs.base_step * 1.001)

    def test_converges_under_skew(self):
        """Producer at nominal 16k, consumer pulling +0.05% fast —
        fill must settle to a stable value, not drain or pin."""
        sr = 44100
        ring = livecap.Ring(sr * 2)
        target = int(0.04 * sr)
        ring.reset(target)
        rs = livecap.Resampler(16000, sr)
        pull = int(round(0.016 * sr * 1.0005))
        fills = []
        for _ in range(4000):               # ~64s of audio
            ring.push(rs.resample(np.full(256, 0.1, np.float32)))
            ring.pull(pull)
            rs.steer(ring.fill() - target, sr)
            fills.append(ring.fill())
        # proportional gain leaves a bounded steady-state offset
        # (~skew*100*sr = 50ms) — the bug was unbounded drain/pin
        self.assertLess(abs(fills[-1] - target), int(0.08 * sr))
        self.assertLess(np.std(fills[-500:]), int(0.005 * sr))
        self.assertLess(
            abs(rs.step - rs.base_step) / rs.base_step, 0.001)


class OpenCableTest(unittest.TestCase):
    """B1 — every failure path must still return a usable cable_sr."""

    def test_empty_candidates(self):
        ring, st, sr = livecap.open_cable(
            [], True, lambda *a: self.fail("open_fn called"))
        self.assertIsNone(ring)
        self.assertIsNone(st)
        self.assertEqual(sr, livecap.SR)

    def test_hidden_cable(self):
        ring, st, sr = livecap.open_cable(
            [(0, 44100, True)], False,
            lambda *a: self.fail("open_fn called"))
        self.assertEqual(sr, livecap.SR)
        self.assertIsNone(st)

    def test_all_candidates_fail(self):
        def bad(*a):
            raise RuntimeError("nope")
        ring, st, sr = livecap.open_cable(
            [(0, 44100, True), (1, 48000, False)], True, bad)
        self.assertIsNone(ring)
        self.assertIsNone(st)
        self.assertEqual(sr, livecap.SR)

    def test_second_candidate_wins(self):
        calls = []

        def fn(idx, sr, wasapi, ring):
            calls.append(idx)
            if idx == 0:
                raise RuntimeError("first broken")
            return "STREAM"
        ring, st, sr = livecap.open_cable(
            [(0, 44100, True), (1, 48000, False)], True, fn)
        self.assertEqual(st, "STREAM")
        self.assertEqual(sr, 48000)
        self.assertEqual(ring.fill(), int(0.04 * 48000))
        self.assertEqual(calls, [0, 1])


class FakePipe:
    """Finals come only from the zero-feed — the B2 regression path."""

    def __init__(self):
        self.partials = {}

    def feed(self, samples, tag):
        if not np.any(samples):
            self.partials.pop(tag, None)
            return "last words"
        self.partials[tag] = "WORDS"
        return None

    def flush(self, tag):
        return None


class SilenceFinalTest(unittest.TestCase):
    """B2 — a final produced by the silence zero-feed must be logged,
    translated and sent to the GUI exactly once."""

    def test_silence_final_is_emitted(self):
        q, gui_q, trans_q = queue.Queue(), queue.Queue(), queue.Queue()
        log, lock = io.StringIO(), threading.Lock()
        t = threading.Thread(
            target=livecap.asr_consumer,
            args=(FakePipe(), q, gui_q, log, lock, "spk", trans_q),
            kwargs=dict(idle_gap=0.05), daemon=True)
        t.start()
        q.put(np.ones(512, np.float32))
        time.sleep(0.5)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        events = []
        while True:
            try:
                events.append(gui_q.get_nowait())
            except queue.Empty:
                break
        finals = [e for e in events if e[0] == "final"]
        self.assertEqual(finals, [("final", "spk", "last words")])
        self.assertIn("[them] last words  [seg 1]", log.getvalue())
        self.assertEqual(trans_q.get_nowait(), (1, "last words"))


class ReporterTest(unittest.TestCase):
    """H1 — one event per transition, nothing on repeats."""

    def test_emits_only_on_change(self):
        q = queue.Queue()
        rep = livecap.Reporter(q)
        rep.set("mic", "ok")
        rep.set("mic", "ok")
        rep.set("mic", "err", "boom")
        rep.set("mic", "err", "boom")
        rep.set("mic", "err", "boom2")
        rep.set("mic", "ok")
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                break
        self.assertEqual(out, [
            ("state", "mic", "ok", ""),
            ("state", "mic", "err", "boom"),
            ("state", "mic", "err", "boom2"),
            ("state", "mic", "ok", "")])


class BadgeTest(unittest.TestCase):
    """H2 — warn/err get badges; ok/info fold into dot + chip."""

    def test_visibility_rule(self):
        self.assertTrue(livecap.badge_shown("warn"))
        self.assertTrue(livecap.badge_shown("err"))
        self.assertFalse(livecap.badge_shown("ok"))
        self.assertFalse(livecap.badge_shown("info"))


class RingResetTest(unittest.TestCase):
    """H4 — unmute/open must leave the jitter prefill in place."""

    def test_reset_prefills(self):
        r = livecap.Ring(1000)
        r.push(np.ones(300, np.float32))
        r.reset(200)
        self.assertEqual(r.fill(), 200)
        self.assertFalse(np.any(r.pull(200)))


class LangCodeTest(unittest.TestCase):
    """M3 — ll-rr inputs keep their region; the whole table."""

    def test_codes(self):
        cases = {
            "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN",
            "zh-CN": "zh-CN", "zh-cn": "zh-CN", "zh-tw": "zh-TW",
            "en": "en-US", "pt": "pt-PT", "pt-BR": "pt-BR",
            "he": "he-IL", "nb": "nb-NB", "xx": "xx-XX",
        }
        for code, want in cases.items():
            with self.subTest(code=code):
                self.assertEqual(livecap.lang_code(code), want)


class PreflightTest(unittest.TestCase):
    """B4 — one missing-path entry per absent file, before sherpa."""

    def _full_tree(self, root, name="zipformer-stream-test"):
        models = root / "models"
        mdir = models / name
        mdir.mkdir(parents=True)
        (mdir / "tokens.txt").write_text("x")
        for p in ("encoder", "decoder", "joiner"):
            (mdir / f"{p}-a.onnx").write_bytes(b"x")
        (models / "gtcrn_simple.onnx").write_bytes(b"x")
        punct = models / livecap.PUNCT_DIR.name
        punct.mkdir()
        (punct / "model.int8.onnx").write_bytes(b"x")
        (punct / "bpe.vocab").write_text("x")
        return mdir

    def test_no_models_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(livecap.preflight(root, None),
                             [root / "models"])

    def test_lists_each_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            mdir = models / "zipformer-stream-x"
            mdir.mkdir(parents=True)
            (mdir / "tokens.txt").write_text("x")
            (mdir / "encoder-a.onnx").write_bytes(b"x")
            names = [str(m) for m in livecap.preflight(root, mdir)]
            for want in ("gtcrn_simple.onnx", "decoder", "joiner",
                         "model.int8.onnx", "bpe.vocab"):
                self.assertTrue(any(want in n for n in names), want)
            self.assertFalse(any("encoder" in n for n in names))

    def test_no_asr_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "models").mkdir()
            (root / "models" / "gtcrn_simple.onnx").write_bytes(b"x")
            names = [str(m) for m in livecap.preflight(root, None)]
            self.assertTrue(any("ASR model dir" in n for n in names))

    def test_nemotron_skips_punct(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            mdir = models / "nemotron-x"
            mdir.mkdir(parents=True)
            (mdir / "tokens.txt").write_text("x")
            for p in ("encoder", "decoder", "joiner"):
                (mdir / f"{p}-a.onnx").write_bytes(b"x")
            (models / "gtcrn_simple.onnx").write_bytes(b"x")
            self.assertEqual(livecap.preflight(root, mdir), [])

    def test_complete_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mdir = self._full_tree(root)
            self.assertEqual(livecap.preflight(root, mdir), [])


class FakeSD:
    """Minimal sounddevice stand-in for pick_mic tests."""

    def __init__(self, devs, default=None, wasapi_default=-1):
        self._devs = devs
        self._wasapi_default = wasapi_default
        self.default = types.SimpleNamespace(device=default or (-1, -1))

    def query_devices(self):
        return self._devs

    def query_hostapis(self):
        return [{"name": "Windows MME", "default_input_device": -1},
                {"name": "Windows WASAPI",
                 "default_input_device": self._wasapi_default}]


def _dev(name, hostapi, inputs=1):
    return {"name": name, "max_input_channels": inputs,
            "hostapi": hostapi}


class PickMicTest(unittest.TestCase):
    """N2 — the WASAPI host API's own default input wins over
    PortAudio's (MME) default; rejected devices still get skipped."""

    def _pick(self, devs, **kw):
        fake = FakeSD(devs, **kw)
        with mock.patch.object(livecap, "sd", fake):
            return livecap.pick_mic()

    def test_wasapi_default_beats_mme_default(self):
        devs = [_dev("Microphone Array (MME)", 0),
                _dev("Microphone (Realtek)", 1)]
        self.assertEqual(
            self._pick(devs, default=(0, 1), wasapi_default=1), 1)

    def test_cable_wasapi_default_skipped(self):
        devs = [_dev("CABLE Output (VB-Audio)", 1),
                _dev("Microphone Array (MME)", 0),
                _dev("Headset Mic (USB)", 1)]
        # wasapi default is the cable -> rejected; MME default is
        # PortAudio's default and wins next
        self.assertEqual(
            self._pick(devs, default=(1, 1), wasapi_default=0), 1)

    def test_no_mic_returns_none(self):
        devs = [_dev("CABLE Output (VB-Audio)", 1),
                _dev("Stereo Mix", 0)]
        self.assertIsNone(
            self._pick(devs, default=(0, 1), wasapi_default=0))


class DiagWriterTest(unittest.TestCase):
    """N3 — a stderr object that raises on write must not stop the
    file from receiving the line."""

    def test_raising_stderr_still_writes_file(self):
        buf = io.StringIO()
        old_log, old_path = livecap._diag_log[0], livecap._diag_path[0]
        old_err = sys.stderr

        class BadErr:
            def write(self, s):
                raise RuntimeError("no console")
        try:
            livecap._diag_log[0] = buf
            livecap._diag_size[0] = 0
            sys.stderr = BadErr()
            livecap.diag("n3-marker-line")
            for _ in range(100):
                if "n3-marker-line" in buf.getvalue():
                    break
                time.sleep(0.02)
            self.assertIn("n3-marker-line", buf.getvalue())
        finally:
            sys.stderr = old_err
            livecap._diag_log[0] = old_log
            livecap._diag_path[0] = old_path


class PlanBadgesTest(unittest.TestCase):
    """N4 — at most one badge, err first, chip hidden under err."""

    def test_chip_only_when_quiet(self):
        lvl = {"trans": "info", "log": "info"}
        txt = {"trans": "ja·online", "log": "log"}
        shown, extra, chip = livecap.plan_badges(lvl, txt)
        self.assertEqual(shown, [])
        self.assertEqual(extra, 0)
        self.assertEqual(chip, "ja·online · log")

    def test_err_shows_badge_hides_chip(self):
        lvl = {"trans": "info", "cable": "err"}
        txt = {"trans": "ja·online", "cable": "failed"}
        shown, extra, chip = livecap.plan_badges(lvl, txt)
        self.assertEqual(shown, ["cable"])
        self.assertEqual(chip, "")

    def test_multiple_problems_counted(self):
        lvl = {"cable": "err", "mic": "warn", "worker": "err"}
        shown, extra, chip = livecap.plan_badges(lvl, {})
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown[0], "cable")   # err before warn,
        self.assertEqual(extra, 2)            # insertion order


class DenoiseFloodTest(unittest.TestCase):
    """N5 — 1000 failures must produce only a few diag lines."""

    def test_bounded(self):
        self.assertLessEqual(
            sum(livecap._should_log_fail(n) for n in range(1, 1001)),
            10)


class FindAsrDirTest(unittest.TestCase):
    """N6 — asr_dir's SystemExit is BaseException: must still yield
    the full preflight list, not crash before it."""

    def test_systemexit_still_preflights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "models").mkdir()
            (root / "models" / "gtcrn_simple.onnx").write_bytes(b"x")
            with mock.patch.object(
                    livecap, "asr_dir",
                    side_effect=SystemExit("no ASR model")):
                self.assertIsNone(livecap.find_asr_dir())
            names = [str(m) for m in livecap.preflight(root, None)]
            self.assertTrue(any("ASR model dir" in n for n in names))


class PipelineLockTest(unittest.TestCase):
    """N7 — every recognizer call runs under the decode lock."""

    def test_all_calls_locked(self):
        locked = {}

        class FakeStream:
            def accept_waveform(self, sr, s):
                pass
            def input_finished(self):
                pass

        class FakeRec:
            def __init__(self):
                self.pipe = None
                self.ready = 1
            def _m(self, name):
                locked[name] = self.pipe.lock.locked()
            def create_stream(self):
                self._m("create_stream")
                return FakeStream()
            def is_ready(self, s):
                self._m("is_ready")
                self.ready -= 1
                return self.ready >= 0
            def decode_stream(self, s):
                self._m("decode_stream")
            def is_endpoint(self, s):
                self._m("is_endpoint")
                return True
            def get_result(self, s):
                self._m("get_result")
                return "TEXT"
            def reset(self, s):
                self._m("reset")

        rec = FakeRec()
        pipe = livecap.Pipeline(rec, None)
        rec.pipe = pipe
        pipe.feed(np.ones(64, np.float32), "spk")
        pipe.flush("spk")
        for name in ("create_stream", "is_ready", "decode_stream",
                     "is_endpoint", "get_result", "reset"):
            self.assertTrue(locked.get(name, False),
                            f"{name} ran unlocked")


class RingCounterTest(unittest.TestCase):
    """N9 — underrun/overrun counters and a stats line that shows
    them."""

    def test_counters(self):
        r = livecap.Ring(100)
        r.pull(10)                                # underrun
        r.push(np.ones(90, np.float32))
        r.push(np.ones(50, np.float32))           # overflow
        self.assertEqual(r.underruns, 1)
        self.assertEqual(r.overruns, 1)

    def test_stat_line(self):
        r = livecap.Ring(100)
        r.pull(5)
        q = queue.Queue(maxsize=1)
        livecap.put_drop(q, "a")
        livecap.put_drop(q, "b")                  # evicts "a"
        mq = queue.Queue(maxsize=2)
        s = livecap.voice_stat(r, q, mq, 2.5, 9.9, 3, 1)
        for want in ("under=1", "drop=1", "max=9.9ms",
                     "slow=3", "fails=1"):
            self.assertIn(want, s)


class PutDropTest(unittest.TestCase):
    """Evictions are counted on the queue and reported."""

    def test_counts_and_returns(self):
        q = queue.Queue(maxsize=1)
        self.assertFalse(livecap.put_drop(q, 1))
        self.assertTrue(livecap.put_drop(q, 2))
        self.assertEqual(q.drops, 1)


class _FakeDen:
    """Echo denoiser — returns input unchanged."""
    frame_shift_in_samples = 256

    def reset(self):
        pass

    def run(self, x, sr):
        return types.SimpleNamespace(samples=np.asarray(x))


class DrainFramesTest(unittest.TestCase):
    """N10 — an oversized host block must drain ALL complete frames,
    not just the first one."""

    def test_all_frames_processed(self):
        mic_q, micasr_q, gui_q = (queue.Queue(), queue.Queue(),
                                  queue.Queue())
        mute, den_off = threading.Event(), threading.Event()
        t = threading.Thread(
            target=livecap.voice_engine,
            args=(mic_q, _FakeDen(), None, micasr_q, mute, den_off,
                  gui_q),
            kwargs=dict(mic_sr=44100), daemon=True)
        t.start()
        mic_q.put(np.zeros(1410, np.float32))   # ~2 frames @16k
        time.sleep(0.4)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        total = 0
        while True:
            try:
                total += len(micasr_q.get_nowait())
            except queue.Empty:
                break
        # 1410 * 16000/44100 = ~511 — the old code emitted ~256 and
        # left the rest pending until the next host block
        self.assertGreaterEqual(total, 500)


class FatalBoxTest(unittest.TestCase):
    """B4 — SystemExit(str) must reach the dialog in the windowed
    build."""

    def test_messagebox_called(self):
        fake = mock.Mock()
        with mock.patch.object(ctypes, "windll", fake):
            livecap._fatal_box("hi")
        fake.user32.MessageBoxW.assert_called_once()


if __name__ == "__main__":
    unittest.main()
