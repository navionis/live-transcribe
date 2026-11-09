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
        rs.steer(10 ** 9, 44100)          # trim + prop both capped
        self.assertAlmostEqual(rs.step, rs.base_step * 1.002)

    def _sim(self, skew, ticks=8000):
        """Run the real Resampler+Ring against a skewed consumer."""
        sr = 44100
        ring = livecap.Ring(sr * 2)
        target = int(0.04 * sr)
        ring.reset(target)
        rs = livecap.Resampler(16000, sr)
        pull = int(round(0.016 * sr * (1 + skew)))
        for _ in range(ticks):
            ring.push(rs.resample(np.full(256, 0.1, np.float32)))
            ring.pull(pull)
            rs.steer(ring.fill() - target, sr)
        return ring, rs, target

    def test_converges_under_skew(self):
        """P5 — integral trim: +-0.05% skew converges inside 10ms of
        the target with zero underruns (proportional-only left a
        ~skew*100s offset)."""
        for skew in (0.0005, -0.0005):
            with self.subTest(skew=skew):
                ring, rs, target = self._sim(skew)
                self.assertLessEqual(abs(ring.fill() - target),
                                     int(0.01 * 44100))
                self.assertEqual(ring.underruns, 0)
                self.assertLessEqual(abs(rs.trim), 0.001)


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
        self.assertEqual(finals, [("final", "spk", "last words", 1)])
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
            for want in ("gtcrn_simple.onnx", "decoder", "joiner"):
                self.assertTrue(any(want in n for n in names), want)
            self.assertFalse(any("encoder" in n for n in names))

    def test_no_asr_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "models").mkdir()
            (root / "models" / "gtcrn_simple.onnx").write_bytes(b"x")
            names = [str(m) for m in livecap.preflight(root, None)]
            self.assertTrue(any("ASR model dir" in n for n in names))

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

    def query_devices(self, idx=None):
        return self._devs if idx is None else self._devs[idx]

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


class DiagFlushTest(unittest.TestCase):
    """P3 — diag_flush drains the queue before teardown, and
    exception hooks log real tracebacks."""

    def _capture(self, fn):
        buf = io.StringIO()
        old = livecap._diag_log[0]
        try:
            livecap._diag_log[0] = buf
            livecap._diag_size[0] = 0
            fn()
            livecap.diag_flush()
        finally:
            livecap._diag_log[0] = old
        return buf.getvalue()

    def test_flush_lands_lines(self):
        out = self._capture(lambda: livecap.diag("p3-marker"))
        self.assertIn("p3-marker", out)

    def test_exc_hook_has_traceback(self):
        def fire():
            try:
                raise KeyError("k")
            except KeyError:
                t, v, tb = sys.exc_info()
                livecap.diag_exc("fatal", t, v, tb)
        out = self._capture(fire)
        self.assertIn("Traceback", out)
        self.assertIn("KeyError", out)


class BadgePriorityTest(unittest.TestCase):
    """P4 — mute must not hide behind another err; order stable."""

    def test_mute_wins_over_cable_err(self):
        shown, extra, _ = livecap.plan_badges(
            {"cable": "err", "mute": "err"}, {})
        self.assertEqual(shown, ["mute"])
        self.assertEqual(extra, 1)

    def test_priority_order_stable(self):
        lvl = {"speaker": "warn", "mic": "warn", "mute": "warn"}
        shown, extra, _ = livecap.plan_badges(lvl, {})
        self.assertEqual(shown, ["mute"])     # warns sort by
        self.assertEqual(extra, 2)            # priority too

    def test_err_beats_mute_warn(self):
        shown, _, _ = livecap.plan_badges(
            {"mute": "warn", "cable": "err"}, {})
        self.assertEqual(shown, ["cable"])    # any err > any warn

    def test_cable_warn_beats_mic_warn(self):
        # Q5: cause beats symptom — "no VB-CABLE" outranks "mic off"
        shown, _, _ = livecap.plan_badges(
            {"mic": "warn", "cable": "warn"}, {})
        self.assertEqual(shown, ["cable"])


class ResetFailTest(unittest.TestCase):
    """P2 — a raising den.reset() must not stick the engine in raw
    bypass mode."""

    def test_engine_recovers(self):
        class Den:
            frame_shift_in_samples = 256
            reset_calls = 0
            run_fails = [True]             # first run() fails once
            def reset(self):
                Den.reset_calls += 1
                raise RuntimeError("reset broke")
            def run(self, x, sr):
                if Den.run_fails and Den.run_fails.pop():
                    raise RuntimeError("run broke")
                return types.SimpleNamespace(samples=np.asarray(x))

        mic_q, micasr_q, gui_q = (queue.Queue(), queue.Queue(),
                                  queue.Queue())
        mute, den_off = threading.Event(), threading.Event()
        t = threading.Thread(
            target=livecap.voice_engine,
            args=(mic_q, Den(), None, micasr_q, mute, den_off, gui_q),
            daemon=True)
        t.start()
        for _ in range(3):
            mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.4)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        states = []
        while True:
            try:
                states.append(gui_q.get_nowait())
            except queue.Empty:
                break
        denoise = [e for e in states
                   if e[0] == "state" and e[1] == "denoise"]
        # reset raised but the flag cleared — engine recovered to ok
        self.assertEqual(denoise[-1], ("state", "denoise", "ok", ""))


class LateRingTest(unittest.TestCase):
    """A cable that reopens at the SAME rate pair must still get an
    rs built — it was None when the engine started with no ring, and
    the rate check alone never rebuilds it."""

    def test_ring_appears_same_rates(self):
        mic_q, micasr_q, gui_q = (queue.Queue(), queue.Queue(),
                                  queue.Queue())
        mute, den_off = threading.Event(), threading.Event()
        state = {"in": None, "out": None, "ring": None,
                 "mic_sr": livecap.SR, "cable_sr": livecap.SR,
                 "mic_dead": False, "cable_dead": False,
                 "ev": threading.Event()}
        t = threading.Thread(
            target=livecap.voice_engine,
            args=(mic_q, _FakeDen(), None, micasr_q, mute, den_off,
                  gui_q),
            kwargs=dict(mic_sr=livecap.SR, cable_sr=livecap.SR,
                        state=state),
            daemon=True)
        t.start()
        mic_q.put(np.zeros(256, np.float32))     # ring still None
        time.sleep(0.1)
        ring = livecap.Ring(livecap.SR // 2)
        state["ring"] = ring                     # cable reopens @16k
        for _ in range(3):
            mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.3)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        mic_errs = []
        while True:
            try:
                e = gui_q.get_nowait()
            except queue.Empty:
                break
            if e[0] == "state" and e[1] == "mic" and e[2] == "err":
                mic_errs.append(e)
        # old code: rs.steer raised AttributeError every chunk ->
        # 'mic engine error' badge, micasr_q starved, ring empty
        self.assertFalse(mic_errs, mic_errs)
        self.assertGreater(ring.fill(), 0)
        self.assertGreater(micasr_q.qsize(), 0)

    def test_ring_swap_same_rates(self):
        """A reopened cable brings a NEW ring object at the same
        rates — the engine must feed it (and rebuild rs for it, so
        the old device's skew estimate can't carry over)."""
        mic_q, micasr_q, gui_q = (queue.Queue(), queue.Queue(),
                                  queue.Queue())
        mute, den_off = threading.Event(), threading.Event()
        ring1 = livecap.Ring(livecap.SR // 2)
        state = {"in": None, "out": None, "ring": ring1,
                 "mic_sr": livecap.SR, "cable_sr": livecap.SR,
                 "mic_dead": False, "cable_dead": False,
                 "ev": threading.Event()}
        t = threading.Thread(
            target=livecap.voice_engine,
            args=(mic_q, _FakeDen(), ring1, micasr_q, mute, den_off,
                  gui_q),
            kwargs=dict(mic_sr=livecap.SR, cable_sr=livecap.SR,
                        state=state),
            daemon=True)
        t.start()
        mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.15)
        ring2 = livecap.Ring(livecap.SR // 2)
        state["ring"] = ring2                # reopen @ same rates
        for _ in range(3):
            mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.3)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        mic_errs = []
        while True:
            try:
                e = gui_q.get_nowait()
            except queue.Empty:
                break
            if e[0] == "state" and e[1] == "mic" and e[2] == "err":
                mic_errs.append(e)
        self.assertFalse(mic_errs, mic_errs)
        self.assertGreater(ring2.fill(), 0)   # new ring is fed


class MicStallTest(unittest.TestCase):
    """P1 — 1.5s of silence on mic_q reports an error; the next chunk
    clears it; mute doesn't suppress."""

    def _run(self, muted):
        mic_q, micasr_q, gui_q = (queue.Queue(), queue.Queue(),
                                  queue.Queue())
        mute, den_off = threading.Event(), threading.Event()
        if muted:
            mute.set()
        t = threading.Thread(
            target=livecap.voice_engine,
            args=(mic_q, _FakeDen(), None, micasr_q, mute, den_off,
                  gui_q),
            kwargs=dict(stall_after=0.15), daemon=True)
        t.start()
        time.sleep(0.35)
        mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.15)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        states = []
        while True:
            try:
                states.append(gui_q.get_nowait())
            except queue.Empty:
                break
        return [e for e in states
                if e[0] == "state" and e[1] == "mic"]

    def test_stall_then_recover(self):
        evs = self._run(muted=False)
        self.assertIn(("state", "mic", "err", "no mic input"), evs)
        self.assertIn(("state", "mic", "ok", ""), evs)

    def test_stall_reported_while_muted(self):
        # Q12: recovery "ok" must clear the badge even while muted
        evs = self._run(muted=True)
        self.assertIn(("state", "mic", "err", "no mic input"), evs)
        self.assertIn(("state", "mic", "ok", ""), evs)


class OpenMicTest(unittest.TestCase):
    """P6 — 16k failure is logged and the native-rate fallback still
    carries auto_convert on WASAPI."""

    def test_fallback_logged_and_wasapi(self):
        calls = []

        class SD(FakeSD):
            class WasapiSettings:
                def __init__(self, **kw):
                    self.kw = kw
            def InputStream(self, **kw):
                calls.append(kw)
                if kw["samplerate"] == 16000:
                    raise RuntimeError("rate rejected")
                return "STREAM"
            def query_devices(self, idx=None):
                return (self._devs if idx is None
                        else self._devs[idx])

        devs = [{"name": "Mic", "max_input_channels": 1,
                 "hostapi": 1, "default_samplerate": 48000}]
        fake = SD(devs)
        den = types.SimpleNamespace(frame_shift_in_samples=256)
        with mock.patch.object(livecap, "sd", fake), \
                mock.patch.object(livecap, "diag") as d:
            st, sr = livecap.open_mic(0, den, queue.Queue(),
                                      queue.Queue())
        self.assertEqual((st, sr), ("STREAM", 48000))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["samplerate"], 48000)
        self.assertIn("extra_settings", calls[1])  # auto_convert kept
        self.assertIn("finished_callback", calls[0])
        self.assertTrue(any("16k open failed" in str(c)
                            for c in d.call_args_list))


class DropToTest(unittest.TestCase):
    """Q1 — a latency burst drops to the target, keeping the newest
    samples; nothing drops inside the band."""

    def test_drops_to_keep(self):
        r = livecap.Ring(1000)
        r.push(np.arange(500, dtype=np.float32))
        dropped = r.drop_to(200)
        self.assertEqual(dropped, 300)
        self.assertEqual(r.fill(), 200)
        self.assertEqual(r.trims, 1)
        self.assertTrue(np.array_equal(r.pull(200),
                                       np.arange(300, 500,
                                                 dtype=np.float32)))

    def test_no_drop_inside_band(self):
        r = livecap.Ring(1000)
        r.push(np.ones(100, np.float32))
        self.assertEqual(r.drop_to(200), 0)
        self.assertEqual(r.trims, 0)


class MaxGapTest(unittest.TestCase):
    """Q2 — pull cadence is recorded as max_gap."""

    def test_records_and_resets(self):
        r = livecap.Ring(1000)
        r.push(np.ones(500, np.float32))
        times = iter([10.0, 10.01, 10.05])
        with mock.patch.object(time, "monotonic",
                               lambda: next(times)):
            r.pull(10)
            r.pull(10)
            r.pull(10)
        self.assertAlmostEqual(r.max_gap, 0.04)
        r.max_gap = 0.0
        self.assertEqual(r.max_gap, 0.0)


class SingleInstanceTest(unittest.TestCase):
    """Q3 — mutex already held -> False; API failure fails open."""

    def test_already_running(self):
        k = mock.Mock()
        k.CreateMutexW.return_value = 7
        with mock.patch.object(ctypes, "WinDLL", return_value=k), \
                mock.patch.object(ctypes, "get_last_error",
                                  return_value=183):
            self.assertFalse(livecap.acquire_single_instance())

    def test_exception_fails_open(self):
        with mock.patch.object(ctypes, "WinDLL",
                               side_effect=OSError("no kernel32")):
            self.assertTrue(livecap.acquire_single_instance())


class SupervisorTest(unittest.TestCase):
    """Q4 — a dead mic is reopened with backoff and the new rate
    lands in state."""

    def test_recovers_after_failures(self):
        ev = threading.Event()
        state = {"in": None, "out": None, "ring": None, "mic_sr": 16000,
                 "cable_sr": 16000, "mic_dead": True, "cable_dead": False,
                 "ev": ev, "cands": []}
        calls = []
        den = types.SimpleNamespace(frame_shift_in_samples=256)

        class St:
            def start(self):
                pass

        def fake_open(i):
            calls.append(i)
            if len(calls) < 3:
                raise RuntimeError("still dead")
            return St(), 48000

        gui_q = queue.Queue()
        t = threading.Thread(
            target=livecap.stream_supervisor,
            args=(state, ev, gui_q, den, queue.Queue()),
            kwargs=dict(open_mic_fn=fake_open, pick_fn=lambda: 0,
                        base=0.01),
            daemon=True)
        t.start()
        time.sleep(0.6)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        self.assertEqual(len(calls), 3)          # failed twice, then won
        self.assertFalse(state["mic_dead"])
        self.assertEqual(state["mic_sr"], 48000)  # new rate reported
        states = [gui_q.get_nowait() for _ in range(gui_q.qsize())]
        self.assertIn(("state", "mic", "ok", ""), states)


class ReinitCooldownTest(unittest.TestCase):
    """One global cooldown — a second _sd_reinit within 30s returns
    False, so mic and cable failures can't bounce each other."""

    def test_second_call_blocked(self):
        sd = mock.Mock()
        state = {"in": None, "out": None, "ring": None,
                 "reinit_t": 0.0, "cands": [], "ev": threading.Event()}
        rep = livecap.Reporter(queue.Queue())
        with mock.patch.object(livecap, "sd", sd), \
                mock.patch.object(livecap, "cable_candidates",
                                  return_value=[(0, 48000, True)]):
            self.assertTrue(livecap._sd_reinit(state, rep))
            self.assertFalse(livecap._sd_reinit(state, rep))
            self.assertEqual(sd._terminate.call_count, 1)
            self.assertEqual(state["cands"], [(0, 48000, True)])


class MicReopenResetTest(unittest.TestCase):
    """A supervisor mic reopen bumps mic_t — the engine must reset the
    denoiser so stale device state can't leak into the new stream."""

    def test_den_reset_on_mic_t_bump(self):
        class Den(_FakeDen):
            resets = [0]
            def reset(self):
                Den.resets[0] += 1

        mic_q, micasr_q, gui_q = (queue.Queue(), queue.Queue(),
                                  queue.Queue())
        mute, den_off = threading.Event(), threading.Event()
        state = {"in": None, "out": None, "ring": None,
                 "mic_sr": livecap.SR, "cable_sr": livecap.SR,
                 "mic_dead": False, "cable_dead": False,
                 "ev": threading.Event()}
        t = threading.Thread(
            target=livecap.voice_engine,
            args=(mic_q, Den(), None, micasr_q, mute, den_off,
                  gui_q),
            kwargs=dict(mic_sr=livecap.SR, cable_sr=livecap.SR,
                        state=state),
            daemon=True)
        t.start()
        mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.2)                    # normal chunk: no reset
        self.assertEqual(Den.resets[0], 0)
        state["mic_t"] = time.time()       # supervisor reopen marker
        time.sleep(0.15)                   # let the drain fire
        mic_q.put(np.zeros(256, np.float32))
        time.sleep(0.2)
        livecap.stop.set()
        t.join(2)
        livecap.stop.clear()
        self.assertEqual(Den.resets[0], 1)


class DeadCbTest(unittest.TestCase):
    """H1 — a replaced stream's finished_callback must not flag its
    successor. PortAudio fires finished_callback on stop()/close()."""

    def _state(self, ev=None):
        return {"in": None, "out": None, "ring": None, "mic_sr": 16000,
                "cable_sr": 16000, "mic_dead": False,
                "cable_dead": False, "ev": ev or threading.Event(),
                "cands": []}

    def test_generation_guard(self):
        st = self._state()
        rep = livecap.Reporter(queue.Queue())
        cb1 = livecap.dead_cb(st, "mic", rep, "died")
        livecap.dead_cb(st, "mic", rep, "died")     # new generation
        cb1()                                     # old stream closed
        self.assertFalse(st["mic_dead"])          # must NOT flag

    def test_reopen_once(self):
        """Fake stream whose stop()/close() fire finished_callback —
        one dead mic => exactly one reopen, mic_dead False at end."""
        st = self._state()
        rep = livecap.Reporter(queue.Queue())
        opens = []

        class FS:
            def __init__(s):
                s.cb = None
            def start(s):
                pass
            def stop(s):
                if s.cb: s.cb()
            def close(s):
                if s.cb: s.cb()

        def fake_open(i):
            opens.append(i)
            f = FS()
            f.cb = livecap.dead_cb(st, "mic", rep, "died")
            return f, 16000

        first, _ = fake_open(0)
        st["in"] = first
        st["mic_dead"] = True
        opens.clear()                      # count supervisor reopens
        t = threading.Thread(
            target=livecap.stream_supervisor,
            args=(st, st["ev"], queue.Queue(), types.SimpleNamespace(
                frame_shift_in_samples=256), queue.Queue()),
            kwargs=dict(open_mic_fn=fake_open, pick_fn=lambda: 0,
                        base=0.01, rep=rep),
            daemon=True)
        t.start()
        time.sleep(0.5)
        livecap.stop.set(); t.join(2); livecap.stop.clear()
        self.assertEqual(len(opens), 1)      # no reopen loop
        self.assertFalse(st["mic_dead"])


class SharedRepTest(unittest.TestCase):
    """H2 — two failure/recovery cycles through mark_dead each end
    with an ok event when the dedupe memory is shared."""

    def test_two_cycles(self):
        gui_q = queue.Queue()
        rep = livecap.Reporter(gui_q)
        st = {"ev": threading.Event(), "mic_dead": False}
        for _ in range(2):
            livecap.mark_dead(st, "mic", rep, "mic stream stopped")
            rep.set("mic", "ok")
            st["mic_dead"] = False
        evs = [gui_q.get_nowait() for _ in range(gui_q.qsize())]
        oks = [e for e in evs if e == ("state", "mic", "ok", "")]
        self.assertEqual(len(oks), 2)


class DampingTest(unittest.TestCase):
    """M4 — a 30ms step disturbance must decay below the deadband,
    not oscillate for minutes (zeta ~0.6)."""

    def test_disturbance_decays(self):
        sr = 44100
        ring = livecap.Ring(sr * 2)
        target = int(0.04 * sr)
        ring.reset(target)
        ring.push(np.zeros(int(0.03 * sr), np.float32))  # +30ms step
        rs = livecap.Resampler(16000, sr)
        pull = int(round(0.016 * sr))
        # ~200s of simulated time at 16ms/tick
        for _ in range(12500):
            ring.push(rs.resample(np.full(256, 0.1, np.float32)))
            ring.pull(pull)
            rs.steer(ring.fill() - target, sr)
        self.assertLessEqual(abs(ring.fill() - target),
                             int(0.005 * sr))   # deadband edge ok


class RingArmedTest(unittest.TestCase):
    """P7 — muted/empty pulls must not count as underruns."""

    def test_clear_disarms(self):
        r = livecap.Ring(100)
        r.pull(10)
        r.push(np.ones(50, np.float32))
        r.clear()
        for _ in range(5):
            r.pull(10)
        self.assertEqual(r.underruns, 0)
        r.push(np.ones(5, np.float32))
        r.pull(10)
        self.assertEqual(r.underruns, 1)


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
        pipe = livecap.Pipeline(rec)
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
        r.push(np.ones(5, np.float32))            # arm the counter
        r.pull(10)                                # underrun
        r.push(np.ones(90, np.float32))
        r.push(np.ones(50, np.float32))           # overflow
        self.assertEqual(r.underruns, 1)
        self.assertEqual(r.overruns, 1)

    def test_stat_line(self):
        r = livecap.Ring(100)
        r.push(np.ones(3, np.float32))
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
