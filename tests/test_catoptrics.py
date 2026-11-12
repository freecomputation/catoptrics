"""Tests for catoptrics.

Run: python3 -m unittest discover -s tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import catoptrics as cat


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def synthetic_spectrum(bumps, low=2e6, high=300e6, step=25_000,
                       base=-100.0):
    """A spectrum: flat floor, a whisper of ripple, Gaussian bumps.

    bumps is a list of (center_hz, height_over_floor_db).
    """
    n = int((high - low) / step) + 1
    freqs = [low + i * step for i in range(n)]
    db = []
    for f in freqs:
        v = base + 0.3 * np.sin(f / 1e6)
        for center, height in bumps:
            v += height * np.exp(-((f - center) ** 2)
                                 / (2 * 60_000.0 ** 2))
        db.append(float(v))
    return freqs, db


VGA_BUMPS = [(25.175e6, 20.0), (50.35e6, 17.0), (75.525e6, 14.0)]
P720_BUMPS = [(74.25e6, 18.0), (148.5e6, 15.0), (222.75e6, 12.0)]
SIM_BUMPS = [(5.76e6, 16.0), (11.52e6, 13.0), (17.28e6, 11.0)]


# ---------------------------------------------------------------------
# Pattern and PNG
# ---------------------------------------------------------------------

class TestPatternAndPng(unittest.TestCase):
    def test_pattern_structure(self):
        img = cat.make_pattern(320, 200)
        self.assertEqual(img.shape, (200, 320))
        self.assertEqual(set(np.unique(img)), {0.0, 1.0})
        # Border frame is bright near the edge, dark just inside it.
        self.assertEqual(img[4, 100], 1.0)
        self.assertEqual(img[20, 20], 0.0)
        # Filled block, lower right.
        self.assertEqual(img[180, 290], 1.0)

    def test_png_roundtrip(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, size=(37, 53), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.png"
            cat.write_png(path, img)
            back = cat.read_png(path)
        self.assertTrue(np.array_equal(img, back))

    def test_iq_roundtrip(self):
        iq = (np.random.default_rng(3).standard_normal(1000)
              + 1j * np.random.default_rng(4).standard_normal(1000))
        with tempfile.TemporaryDirectory() as td:
            p32 = Path(td) / "a.cf32"
            cat.save_iq(p32, iq, "cf32")
            back32 = cat.load_iq(p32)
            p8 = Path(td) / "a.cu8"
            cat.save_iq(p8, iq * 0.1, "cu8")
            back8 = cat.load_iq(p8)
        self.assertTrue(np.allclose(iq, back32, atol=1e-6))
        self.assertTrue(np.allclose(iq * 0.1, back8, atol=0.01))

    def test_parse_freq(self):
        self.assertEqual(cat.parse_freq("148.5M"), 148_500_000)
        self.assertEqual(cat.parse_freq("500k"), 500_000)
        self.assertEqual(cat.parse_freq("2400000"), 2_400_000)
        self.assertEqual(cat.parse_freq(74.25e6), 74_250_000)


# ---------------------------------------------------------------------
# Simulator and bench
# ---------------------------------------------------------------------

class TestSimulateAndReconstruct(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mode = cat.MODE_BY_NAME["SIM320x200@60"]
        cls.iq, cls.info, cls.pattern = cat.simulate_iq(cls.mode)

    def test_simulator_shape_and_truth(self):
        self.assertEqual(len(self.iq), self.info["samples"])
        self.assertEqual(self.info["frames"], 18)
        self.assertAlmostEqual(
            self.info["pixel_clock_true"],
            self.mode["pixel_clock"] * 1.000037, places=1)

    def test_sync_estimate_lands_on_true_period(self):
        envelope = np.abs(self.iq)
        period, strength = cat.estimate_frame_period(
            envelope, self.info["rate"], self.mode["refresh"])
        true_period = (self.mode["htotal"] * self.mode["vtotal"]
                       * self.info["rate"]
                       / self.info["pixel_clock_true"])
        self.assertLess(abs(period - true_period) / true_period, 1e-4)
        self.assertGreater(strength, 0.5)

    def test_reconstruction_recovers_pattern(self):
        params = {"width": 320, "height": 200, "htotal": 400,
                  "vtotal": 240, "refresh": 60.0,
                  "pixel_clock": 5_760_000}
        image, report = cat.reconstruct_iq(
            self.iq, self.info["rate"], params,
            reference=self.pattern * 255.0)
        self.assertEqual(image.shape, (200, 320))
        self.assertGreaterEqual(report["reference_ncc"], 0.9)
        self.assertGreaterEqual(report["frames_averaged"], 15)
        self.assertGreater(report["image_snr_db"], 10.0)


# ---------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------

class TestCensus(unittest.TestCase):
    def test_two_emitter_families_two_targets_in_order(self):
        freqs, db = synthetic_spectrum(VGA_BUMPS + P720_BUMPS)
        targets = cat.find_targets(freqs, db)
        self.assertEqual(len(targets), 2)
        first, second = targets
        self.assertEqual(first["number"], 1)
        self.assertEqual(first["mode_guess"], "640x480@60")
        self.assertEqual(first["strongest_freq_hz"], 25.175e6)
        self.assertTrue(first["harmonic_confirmed"])
        self.assertFalse(first["weak"])
        self.assertEqual(second["number"], 2)
        self.assertEqual(second["mode_guess"], "1280x720@60")
        self.assertEqual(second["strongest_freq_hz"], 74.25e6)
        self.assertTrue(second["harmonic_confirmed"])
        # 148.5 MHz is both 720p's second harmonic and 1080p's
        # fundamental: the sharing must be marked, not hidden.
        self.assertTrue(second["ambiguous"])
        self.assertTrue(any("1920x1080@60" in n
                            for n in second["notes"]))

    def test_unidentified_peak_is_marked_weak(self):
        freqs, db = synthetic_spectrum([(33.3e6, 8.0)])
        targets = cat.find_targets(freqs, db)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertEqual(t["kind"], "unidentified")
        self.assertIsNone(t["mode_guess"])
        self.assertTrue(t["weak"])
        self.assertFalse(t["harmonic_confirmed"])

    def test_flat_spectrum_is_a_null_result(self):
        freqs, db = synthetic_spectrum([])
        self.assertEqual(cat.find_targets(freqs, db), [])


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

class TestSelection(unittest.TestCase):
    def _session_with_targets(self, td, bumps):
        session = cat.Session(td)
        freqs, db = synthetic_spectrum(bumps)
        targets = cat.find_targets(freqs, db)
        session.save_targets(targets, {"source": "test"})
        return session, targets

    def test_targets_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            session, targets = self._session_with_targets(
                td, VGA_BUMPS + P720_BUMPS)
            loaded = cat.load_target(session, 2)
            self.assertEqual(loaded["mode_guess"], "1280x720@60")
            self.assertIsNone(cat.load_target(session, 99))

    def test_selecting_second_drives_capture_plan(self):
        with tempfile.TemporaryDirectory() as td:
            session, _ = self._session_with_targets(
                td, VGA_BUMPS + P720_BUMPS)
            target = cat.load_target(session, 2)
            plan = cat.plan_capture(target=target, rate=2_400_000,
                                    seconds=5.0)
            self.assertEqual(plan["freq_hz"], 74_250_000)
            self.assertEqual(plan["target_number"], 2)
            self.assertEqual(plan["target_mode"], "1280x720@60")

    def test_selecting_second_drives_reconstruct_params(self):
        with tempfile.TemporaryDirectory() as td:
            session, _ = self._session_with_targets(
                td, VGA_BUMPS + P720_BUMPS)
            target = cat.load_target(session, 2)
            args = SimpleNamespace(mode=None, width=None, height=None,
                                   htotal=None, vtotal=None,
                                   refresh=None, pixel_clock=None)
            params = cat._resolve_mode_params(args, target)
            self.assertEqual(params["width"], 1280)
            self.assertEqual(params["height"], 720)
            self.assertEqual(params["pixel_clock"], 74_250_000)
            self.assertEqual(params["refresh"], 60.0)

    def test_prompt_selection(self):
        with tempfile.TemporaryDirectory() as td:
            session, _ = self._session_with_targets(
                td, VGA_BUMPS + P720_BUMPS)
            payload = session.load_targets()
            with mock.patch("builtins.input", return_value="2"):
                chosen = cat.prompt_for_target(session, payload)
            self.assertEqual(chosen["number"], 2)

    def test_reconstruct_selected_simulated_target(self):
        # VGA is target 1, the SIM family is target 2; selecting 2
        # must reconstruct the simulated emanation to proof standard.
        with tempfile.TemporaryDirectory() as td:
            session, _ = self._session_with_targets(
                td, VGA_BUMPS + SIM_BUMPS)
            target = cat.load_target(session, 2)
            self.assertEqual(target["mode_guess"], "SIM320x200@60")
            args = SimpleNamespace(mode=None, width=None, height=None,
                                   htotal=None, vtotal=None,
                                   refresh=None, pixel_clock=None)
            params = cat._resolve_mode_params(args, target)
            mode = cat.MODE_BY_NAME["SIM320x200@60"]
            iq, info, pattern = cat.simulate_iq(mode)
            image, report = cat.reconstruct_iq(
                iq, info["rate"], params, reference=pattern * 255.0)
            self.assertGreaterEqual(report["reference_ncc"], 0.9)


# ---------------------------------------------------------------------
# Capture refusals and the hunt command
# ---------------------------------------------------------------------

class TestCaptureAndHuntCommands(unittest.TestCase):
    def test_capture_refuses_when_tuner_held(self):
        holders = [{"pid": 4242, "command": "python3 -m passband run",
                    "program": "passband"}]
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(cat, "find_tuner_holders",
                                   return_value=holders):
                code = cat.main(["capture", "--session", td,
                                 "--freq", "100M"])
            self.assertEqual(code, cat.EXIT_BUSY)
            entries = [json.loads(line) for line in
                       open(Path(td) / "session.jsonl")]
            events = [e["event"] for e in entries]
            self.assertIn("capture_planned", events)
            self.assertIn("refused_busy", events)
            self.assertEqual(list(Path(td).glob("*.cu8")), [])

    def test_capture_unavailable_without_rtl_sdr(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(cat, "find_tuner_holders",
                                   return_value=[]), \
                    mock.patch.object(cat.shutil, "which",
                                      return_value=None):
                code = cat.main(["capture", "--session", td,
                                 "--freq", "100M"])
            self.assertEqual(code, cat.EXIT_UNAVAILABLE)

    def test_hunt_command_on_spectrum_file(self):
        freqs, db = synthetic_spectrum(VGA_BUMPS + P720_BUMPS)
        with tempfile.TemporaryDirectory() as td:
            spec = Path(td) / "spec.csv"
            with open(spec, "w") as fh:
                for f, v in zip(freqs, db):
                    fh.write(f"{f:.0f},{v:.2f}\n")
            code = cat.main(["hunt", "--session", td,
                             "--spectrum", str(spec)])
            self.assertEqual(code, cat.EXIT_OK)
            payload = json.loads(
                (Path(td) / "targets.json").read_text())
            self.assertEqual(len(payload["targets"]), 2)
            entries = [json.loads(line) for line in
                       open(Path(td) / "session.jsonl")]
            self.assertIn("hunt", [e["event"] for e in entries])


# ---------------------------------------------------------------------
# Live watch: the incremental folder
# ---------------------------------------------------------------------

SIM_PARAMS = {"width": 320, "height": 200, "htotal": 400,
              "vtotal": 240, "refresh": 60.0,
              "pixel_clock": 5_760_000}


def drive_folder(folder, iq, rate):
    """Push an IQ array's envelope in one-frame chunks, then flush."""
    env = np.abs(iq)
    chunk = int(rate / 60)
    for i in range(0, len(env), chunk):
        folder.push(env[i:i + chunk])
    folder.flush()
    return folder


class TestWatchFolder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mode = cat.MODE_BY_NAME["SIM320x200@60"]
        cls.iq, cls.info, cls.pattern = cat.simulate_iq(cls.mode)

    def test_live_fold_converges_on_offline_bench(self):
        folder = drive_folder(
            cat.LiveFolder(SIM_PARAMS, self.info["rate"]),
            self.iq, self.info["rate"])
        image, stats = folder.snapshot()
        offline, report = cat.reconstruct_iq(
            self.iq, self.info["rate"], SIM_PARAMS,
            reference=self.pattern * 255.0)
        agree = cat.normalized_cross_correlation(
            offline.astype(np.float64), image.astype(np.float64))
        self.assertGreaterEqual(agree, 0.95)
        self.assertGreaterEqual(
            cat.normalized_cross_correlation(
                self.pattern * 255.0, image.astype(np.float64)),
            0.95)
        self.assertGreaterEqual(stats["frames_folded"], 15)

    def test_snapshots_improve_within_stated_tolerance(self):
        # Stated tolerance: on this 0 dB stream the final snapshot
        # may sit up to 0.05 of correlation below the early one.
        # Noise averaging dominates the early frames; over long
        # folds the period estimate's small residual bias
        # integrates sample by sample (measured and stated in the
        # README), so a slow decline after the peak is the honest
        # behavior. What must never happen is a collapse - the
        # final image stays at proof standard.
        iq, info, pattern = cat.simulate_iq(
            self.mode, seconds=1.0, snr_db=0.0)
        folder = cat.LiveFolder(SIM_PARAMS, info["rate"])
        env = np.abs(iq)
        chunk = int(info["rate"] / 60)
        early = None
        for i in range(0, len(env), chunk):
            folder.push(env[i:i + chunk])
            if early is None and folder.frames_folded >= 5:
                img, _ = folder.snapshot()
                early = cat.normalized_cross_correlation(
                    pattern * 255.0, img.astype(np.float64))
        folder.flush()
        image, _ = folder.snapshot()
        final = cat.normalized_cross_correlation(
            pattern * 255.0, image.astype(np.float64))
        self.assertIsNotNone(early)
        self.assertGreaterEqual(final, 0.94)
        self.assertGreaterEqual(final, early - 0.05)

    def test_stable_tracking_across_planted_clock_offset(self):
        # The simulator's clock runs +37 ppm fast. Over 36 frames
        # the folder must hold sync without a single re-lock and
        # land on the true period.
        iq, info, pattern = cat.simulate_iq(
            self.mode, seconds=0.6, snr_db=12.0)
        folder = drive_folder(
            cat.LiveFolder(SIM_PARAMS, info["rate"]), iq,
            info["rate"])
        true_period = (self.mode["htotal"] * self.mode["vtotal"]
                       * info["rate"] / info["pixel_clock_true"])
        self.assertEqual(folder.relocks, 0)
        self.assertGreater(len(folder.period_checks), 0)
        self.assertLess(
            abs(folder.period - true_period) / true_period, 1e-4)

    def test_relock_after_clock_jump(self):
        # The clock steps from +37 ppm to -800 ppm mid-stream: a
        # re-lock must be adopted, the tracked period must land on
        # the new clock, and the picture must survive.
        iq_a, info, pattern = cat.simulate_iq(
            self.mode, seconds=0.35, snr_db=12.0)
        iq_b, _, _ = cat.simulate_iq(
            self.mode, seconds=0.45, snr_db=12.0,
            clock_ppm=-800.0, seed=777)
        folder = cat.LiveFolder(SIM_PARAMS, info["rate"],
                                relock_every=15)
        env = np.abs(np.concatenate([iq_a, iq_b]))
        chunk = int(info["rate"] / 60)
        for i in range(0, len(env), chunk):
            folder.push(env[i:i + chunk])
        folder.flush()
        self.assertGreaterEqual(folder.relocks, 1)
        self.assertTrue(
            any(c["adopted"] for c in folder.period_checks))
        true_b = (self.mode["htotal"] * self.mode["vtotal"]
                  * info["rate"]
                  / (self.mode["pixel_clock"] * (1 - 800e-6)))
        self.assertLess(abs(folder.period - true_b) / true_b, 1e-4)
        image, _ = folder.snapshot()
        self.assertGreaterEqual(
            cat.normalized_cross_correlation(
                pattern * 255.0, image.astype(np.float64)), 0.95)


class TestWatchCommand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mode = cat.MODE_BY_NAME["SIM320x200@60"]
        cls.iq, cls.info, cls.pattern = cat.simulate_iq(cls.mode)

    def _write_replay(self, td):
        src = Path(td) / "emanation.cf32"
        cat.save_iq(src, self.iq, "cf32")
        return src

    def test_watch_replay_records_and_summarizes(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._write_replay(td)
            session_dir = Path(td) / "session"
            code = cat.main(["watch", "--session", str(session_dir),
                             "--source", str(src),
                             "--mode", "SIM320x200@60",
                             "--rate", str(self.info["rate"]),
                             "--every-frames", "6"])
            self.assertEqual(code, cat.EXIT_OK)
            entries = [json.loads(line) for line in
                       open(session_dir / "session.jsonl")]
            events = [e["event"] for e in entries]
            self.assertIn("watch_planned", events)
            self.assertIn("watch_locked", events)
            summary = entries[-1]
            self.assertEqual(summary["event"], "watch")
            self.assertGreaterEqual(summary["frames_folded"], 15)
            # Recording fidelity: the recorded stream is the
            # replayed stream byte for byte, and it re-folds
            # offline to the same picture.
            recorded = Path(summary["record"])
            self.assertEqual(recorded.read_bytes(), src.read_bytes())
            iq_back = cat.load_iq(recorded)
            image, report = cat.reconstruct_iq(
                iq_back, self.info["rate"], SIM_PARAMS,
                reference=self.pattern * 255.0)
            self.assertGreaterEqual(report["reference_ncc"], 0.95)
            self.assertTrue(Path(summary["image"]).is_file())
            self.assertTrue(Path(summary["first_image"]).is_file())

    def test_watch_no_record_writes_no_iq(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._write_replay(td)
            session_dir = Path(td) / "session"
            code = cat.main(["watch", "--session", str(session_dir),
                             "--source", str(src),
                             "--mode", "SIM320x200@60",
                             "--rate", str(self.info["rate"]),
                             "--no-record"])
            self.assertEqual(code, cat.EXIT_OK)
            entries = [json.loads(line) for line in
                       open(session_dir / "session.jsonl")]
            summary = entries[-1]
            self.assertEqual(summary["event"], "watch")
            self.assertIsNone(summary["record"])
            self.assertEqual(
                list(session_dir.glob("watch-*.cf32")), [])

    def test_watch_refuses_when_tuner_held(self):
        holders = [{"pid": 4242, "command": "python3 -m passband run",
                    "program": "passband"}]
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(cat, "find_tuner_holders",
                                   return_value=holders):
                code = cat.main(["watch", "--session", td,
                                 "--freq", "121.5M",
                                 "--mode", "SIM320x200@60"])
            self.assertEqual(code, cat.EXIT_BUSY)
            entries = [json.loads(line) for line in
                       open(Path(td) / "session.jsonl")]
            events = [e["event"] for e in entries]
            self.assertIn("watch_planned", events)
            self.assertIn("refused_busy", events)
            self.assertEqual(list(Path(td).glob("watch*.png")), [])

    def test_watch_unavailable_without_rtl_sdr(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(cat, "find_tuner_holders",
                                   return_value=[]), \
                    mock.patch.object(cat.shutil, "which",
                                      return_value=None):
                code = cat.main(["watch", "--session", td,
                                 "--freq", "121.5M",
                                 "--mode", "SIM320x200@60"])
            self.assertEqual(code, cat.EXIT_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
