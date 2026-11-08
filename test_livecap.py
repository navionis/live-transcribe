#!/usr/bin/env python
"""Unit tests for livecap — pure logic only, no audio devices needed.

Run: .venv\\Scripts\\python.exe -m unittest test_livecap -v
"""

import ctypes
import io
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import livecap


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
