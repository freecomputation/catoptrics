#!/usr/bin/env python3
"""Catoptrics - a Van Eck emanation viewer.

A computer display and its cable radiate a faint radio copy of the
image being shown. Catoptrics receives that radiation with an
RTL-SDR (RTL2832U family) and reconstructs the picture from it.
Receive-only: it transmits nothing and reads a signal that is
already in the air.

The workflow is a census and a selection:

    hunt        sweep a band, group peaks into emanation sources,
                and print a numbered target list (targets.json)
    capture     record raw IQ, for a selected target or a frequency
    watch       the live bench: fold frames from a stream as they
                arrive, writing a picture that sharpens while you
                look at it, and recording the raw IQ as it goes
    reconstruct fold a capture back into a raster image, with a
                written report of sync quality and image SNR
    simulate    synthesize a raster emanation from a built-in test
                pattern, to prove the bench end to end

Every claim the tool makes is a number it printed: levels are
measured against a local noise floor, sync quality is an
autocorrelation strength, and a reconstruction is scored against a
known pattern when one is supplied. A null result is reported as a
null result.

numpy is required for capture reconstruction and simulation; the
census itself is standard library. A command whose dependency is
absent reports unavailable rather than crashing.

Exit codes: 0 ok, 2 unavailable or usage problem, 3 tuner busy.

Usage:
    python3 catoptrics.py hunt --low 40M --high 400M
    python3 catoptrics.py capture --target 2 --seconds 5
    python3 catoptrics.py watch --target 2
    python3 catoptrics.py reconstruct --iq capture.cu8 --target 2
    python3 catoptrics.py simulate --out sim.iq
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None
    HAS_NUMPY = False

__version__ = "0.2.0"

EXIT_OK = 0
EXIT_UNAVAILABLE = 2
EXIT_BUSY = 3

# ---------------------------------------------------------------------
# Display modes. Totals include blanking: a frame is htotal by vtotal
# pixel slots, of which width by height are visible. The pixel clock
# is htotal * vtotal * refresh. The last entry is the simulator's
# native test mode, small enough to synthesize and fold quickly.
# ---------------------------------------------------------------------

MODES = [
    {"name": "640x480@60", "width": 640, "height": 480,
     "htotal": 800, "vtotal": 525, "pixel_clock": 25_175_000,
     "refresh": 59.94},
    {"name": "1024x768@60", "width": 1024, "height": 768,
     "htotal": 1344, "vtotal": 806, "pixel_clock": 65_000_000,
     "refresh": 60.0},
    {"name": "1280x720@60", "width": 1280, "height": 720,
     "htotal": 1650, "vtotal": 750, "pixel_clock": 74_250_000,
     "refresh": 60.0},
    {"name": "1920x1080@60", "width": 1920, "height": 1080,
     "htotal": 2200, "vtotal": 1125, "pixel_clock": 148_500_000,
     "refresh": 60.0},
    {"name": "SIM320x200@60", "width": 320, "height": 200,
     "htotal": 400, "vtotal": 240, "pixel_clock": 5_760_000,
     "refresh": 60.0},
]

MODE_BY_NAME = {m["name"]: m for m in MODES}

TUNER_PROGRAMS = (
    "rtl_sdr", "rtl_power", "rtl_fm", "rtl_433", "readsb",
    "dumpvdl2", "acarsdec", "rtl_airband", "passband", "satdump",
    "ais-catcher", "multimon-ng",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_freq(text) -> int:
    """Parse a frequency: plain Hz, or with a k/M/G suffix."""
    if isinstance(text, (int, float)):
        return int(text)
    s = str(text).strip()
    mult = 1
    if s and s[-1] in "kKMG":
        mult = {"k": 1_000, "K": 1_000, "M": 1_000_000,
                "G": 1_000_000_000}[s[-1]]
        s = s[:-1]
    return int(round(float(s) * mult))


def fmt_freq(hz) -> str:
    if hz is None:
        return "-"
    if hz >= 1_000_000:
        return f"{hz / 1e6:.3f} MHz"
    if hz >= 1_000:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz} Hz"


# ---------------------------------------------------------------------
# The session: a directory holding the append-only session log, the
# persisted target list, and the captures. Every command records
# what it did, with its parameters, in session.jsonl.
# ---------------------------------------------------------------------

class Session:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "session.jsonl"
        self.targets_path = self.root / "targets.json"

    def log(self, event: str, **fields):
        entry = {"ts": now_iso(), "event": event}
        entry.update(fields)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def save_targets(self, targets, hunt_info):
        payload = {"saved": now_iso(), "hunt": hunt_info,
                   "targets": targets}
        with open(self.targets_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        return payload

    def load_targets(self):
        if not self.targets_path.is_file():
            return None
        with open(self.targets_path, encoding="utf-8") as fh:
            return json.load(fh)


# ---------------------------------------------------------------------
# Spectrum input. Two formats: rtl_power CSV (one sweep per line:
# date, time, hz_low, hz_high, hz_step, num_samples, then dB bins)
# and a plain two-column text file of frequency_Hz, dB pairs.
# ---------------------------------------------------------------------

def parse_rtl_power_csv(text: str):
    freqs, db = [], []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            low = float(parts[2])
            step = float(parts[4])
            values = [float(v) for v in parts[6:] if v != ""]
        except ValueError:
            continue
        for i, v in enumerate(values):
            freqs.append(low + i * step)
            db.append(v)
    return freqs, db


def parse_simple_spectrum(text: str):
    freqs, db = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            freqs.append(float(parts[0]))
            db.append(float(parts[1]))
        except ValueError:
            continue
    return freqs, db


def load_spectrum_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    freqs, db = parse_rtl_power_csv(text)
    if not freqs:
        freqs, db = parse_simple_spectrum(text)
    order = sorted(range(len(freqs)), key=lambda i: freqs[i])
    return [freqs[i] for i in order], [db[i] for i in order]


# ---------------------------------------------------------------------
# Peak detection and the census.
#
# The floor under a bin is the median of a wide window around it, so
# a peak's level over floor is a local, honest number. Peaks are
# local maxima at least threshold_db over that floor.
# ---------------------------------------------------------------------

def local_floor(db, window=51):
    n = len(db)
    half = window // 2
    floor = []
    if HAS_NUMPY:
        arr = np.asarray(db, dtype=float)
        padded = np.pad(arr, half, mode="edge")
        # Sliding median via stride tricks; window is small.
        from numpy.lib.stride_tricks import sliding_window_view
        floor = list(np.median(sliding_window_view(padded, window), axis=1))
        return [float(v) for v in floor]
    import statistics
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        floor.append(float(statistics.median(db[lo:hi])))
    return floor


def find_peaks(freqs, db, threshold_db=6.0, min_separation_bins=3):
    floor = local_floor(db)
    candidates = []
    for i in range(len(db)):
        over = db[i] - floor[i]
        if over < threshold_db:
            continue
        if i > 0 and db[i] < db[i - 1]:
            continue
        if i < len(db) - 1 and db[i] < db[i + 1]:
            continue
        candidates.append((over, i))
    candidates.sort(reverse=True)
    chosen = []
    for over, i in candidates:
        if all(abs(i - j) >= min_separation_bins for _o, j in chosen):
            chosen.append((over, i))
    peaks = []
    for over, i in sorted(chosen, key=lambda t: t[1]):
        peaks.append({"freq_hz": float(freqs[i]), "level_db": float(db[i]),
                      "floor_db": float(floor[i]),
                      "over_floor_db": float(over), "bin": i})
    return peaks


def find_targets(freqs, db, modes=MODES, threshold_db=6.0,
                 weak_db=12.0, max_harmonic=8):
    """Group peaks into emanation-source targets.

    A family is one mode's pixel clock and its harmonics. Families
    are scored by their total evidence and claim peaks greedily,
    strongest family first, so a peak shared between two possible
    families (the second harmonic of 1280x720 is the fundamental of
    1920x1080) is counted once and the sharing is noted as ambiguity.
    Peaks no family claims become unidentified single-peak targets.
    """
    peaks = find_peaks(freqs, db, threshold_db=threshold_db)
    if not peaks:
        return []
    bin_step = (freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0

    families = []
    for mode in modes:
        fpix = float(mode["pixel_clock"])
        matched = []
        for k in range(1, max_harmonic + 1):
            target_f = fpix * k
            tol = max(0.004 * target_f, 2.0 * bin_step)
            best = None
            for peak in peaks:
                if abs(peak["freq_hz"] - target_f) <= tol:
                    if best is None or peak["over_floor_db"] > \
                            best["over_floor_db"]:
                        best = peak
            if best is not None:
                matched.append({"harmonic": k, "peak": best})
        if matched:
            families.append({
                "mode": mode, "matched": matched,
                "score": sum(m["peak"]["over_floor_db"] for m in matched),
            })

    families.sort(key=lambda f: -f["score"])
    claimed = {}   # peak bin -> family index that claimed it
    contested = {}  # peak bin -> list of mode names that matched it
    for fam in families:
        for m in fam["matched"]:
            contested.setdefault(m["peak"]["bin"], []).append(
                fam["mode"]["name"])
    surviving = []
    for idx, fam in enumerate(families):
        mine = []
        for m in fam["matched"]:
            b = m["peak"]["bin"]
            if b not in claimed:
                claimed[b] = idx
                mine.append(m)
        if mine:
            surviving.append((fam, mine))

    targets = []
    for fam, mine in surviving:
        mode = fam["mode"]
        strongest = max(mine, key=lambda m: m["peak"]["over_floor_db"])
        notes = []
        ambiguous = False
        for m in mine:
            others = [n for n in contested[m["peak"]["bin"]]
                      if n != mode["name"]]
            if others:
                ambiguous = True
                notes.append(
                    f"peak at {fmt_freq(m['peak']['freq_hz'])} also "
                    f"matches {', '.join(others)}")
        confirmed = len(mine) >= 2
        weak = strongest["peak"]["over_floor_db"] < weak_db or not confirmed
        if not confirmed:
            notes.append("single harmonic; no second-harmonic "
                         "confirmation")
        targets.append({
            "kind": "mode",
            "mode_guess": mode["name"],
            "width": mode["width"], "height": mode["height"],
            "htotal": mode["htotal"], "vtotal": mode["vtotal"],
            "pixel_clock": mode["pixel_clock"],
            "refresh_est": mode["refresh"],
            "strongest_freq_hz": strongest["peak"]["freq_hz"],
            "strongest_over_floor_db":
                strongest["peak"]["over_floor_db"],
            "peaks": [{"freq_hz": m["peak"]["freq_hz"],
                       "level_db": m["peak"]["level_db"],
                       "over_floor_db": m["peak"]["over_floor_db"],
                       "harmonic": m["harmonic"]} for m in mine],
            "harmonic_confirmed": confirmed,
            "weak": bool(weak), "ambiguous": bool(ambiguous),
            "notes": notes,
        })

    claimed_bins = set(claimed)
    for peak in peaks:
        if peak["bin"] in claimed_bins:
            continue
        targets.append({
            "kind": "unidentified",
            "mode_guess": None, "width": None, "height": None,
            "htotal": None, "vtotal": None, "pixel_clock": None,
            "refresh_est": None,
            "strongest_freq_hz": peak["freq_hz"],
            "strongest_over_floor_db": peak["over_floor_db"],
            "peaks": [{"freq_hz": peak["freq_hz"],
                       "level_db": peak["level_db"],
                       "over_floor_db": peak["over_floor_db"],
                       "harmonic": None}],
            "harmonic_confirmed": False,
            "weak": bool(peak["over_floor_db"] < weak_db),
            "ambiguous": False,
            "notes": ["no pixel-clock family match"],
        })

    targets.sort(key=lambda t: -t["strongest_over_floor_db"])
    for i, t in enumerate(targets, start=1):
        t["number"] = i
    return targets


def format_target_list(targets):
    lines = []
    if not targets:
        lines.append("no targets: no peak stood far enough over its "
                     "local floor")
        return lines
    for t in targets:
        flags = []
        if t["weak"]:
            flags.append("weak")
        if t["ambiguous"]:
            flags.append("ambiguous")
        flag_text = f"  [{', '.join(flags)}]" if flags else ""
        mode = t["mode_guess"] or "unknown mode"
        refresh = (f"{t['refresh_est']:.2f} Hz" if t["refresh_est"]
                   else "refresh unknown")
        lines.append(
            f"{t['number']:>2}. {mode}  strongest "
            f"{fmt_freq(t['strongest_freq_hz'])} at "
            f"{t['strongest_over_floor_db']:.1f} dB over floor, "
            f"{refresh}{flag_text}")
        for p in t["peaks"]:
            harm = (f" (harmonic {p['harmonic']})"
                    if p["harmonic"] else "")
            lines.append(
                f"      {fmt_freq(p['freq_hz'])}  "
                f"{p['level_db']:.1f} dB, "
                f"{p['over_floor_db']:.1f} over floor{harm}")
        for note in t["notes"]:
            lines.append(f"      note: {note}")
    return lines


# ---------------------------------------------------------------------
# Tuner tenancy. The receiver is shared property: before any command
# touches it, the process table is scanned for programs known to
# hold one. A holder means refusal, with the holder named - never a
# kill, never a preemption.
# ---------------------------------------------------------------------

def find_tuner_holders():
    holders = []
    proc = Path("/proc")
    if not proc.is_dir():
        return holders
    me = str(Path(f"/proc/self").resolve()).split("/")[-1]
    for entry in proc.iterdir():
        if not entry.name.isdigit() or entry.name == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if not raw:
            continue
        cmdline = raw.replace(b"\0", b" ").decode(
            "utf-8", errors="replace").strip()
        lowered = cmdline.lower()
        for prog in TUNER_PROGRAMS:
            if prog in lowered:
                holders.append({"pid": int(entry.name),
                                "command": cmdline,
                                "program": prog})
                break
    return holders


def refuse_if_busy(session, action):
    """Returns a holders list; the caller refuses if it is non-empty."""
    holders = find_tuner_holders()
    if holders:
        session.log("refused_busy", action=action, holders=holders)
    return holders


def print_busy(holders):
    for h in holders:
        print(f"tuner busy: held by pid {h['pid']} ({h['command']})",
              file=sys.stderr)
    print("catoptrics does not preempt a running program; "
          "try again when the tuner is free", file=sys.stderr)


# ---------------------------------------------------------------------
# IQ storage. Captures from rtl_sdr are unsigned 8-bit I/Q (.cu8).
# The simulator writes 32-bit float I/Q (.cf32), cheaper to make
# exact. The loader accepts both, by extension or forced format.
# ---------------------------------------------------------------------

def save_iq(path, iq, fmt):
    path = Path(path)
    if fmt == "cf32":
        arr = np.asarray(iq, dtype=np.complex64)
        interleaved = np.empty(arr.size * 2, dtype=np.float32)
        interleaved[0::2] = arr.real
        interleaved[1::2] = arr.imag
        interleaved.tofile(path)
    elif fmt == "cu8":
        arr = np.asarray(iq)
        # Scale the channels separately: numpy's clip on a complex
        # array does not clip real and imaginary parts independently.
        i_chan = np.clip(np.round(arr.real * 127.4 + 127.5), 0, 255)
        q_chan = np.clip(np.round(arr.imag * 127.4 + 127.5), 0, 255)
        out = np.empty(arr.size * 2, dtype=np.uint8)
        out[0::2] = i_chan.astype(np.uint8)
        out[1::2] = q_chan.astype(np.uint8)
        out.tofile(path)
    else:
        raise ValueError(f"unknown IQ format {fmt}")
    return path


def load_iq(path, fmt=None):
    path = Path(path)
    if fmt is None:
        fmt = "cf32" if path.suffix.lower() in (".cf32", ".iq") \
            else "cu8"
    if fmt == "cf32":
        raw = np.fromfile(path, dtype=np.float32)
        raw = raw[: raw.size // 2 * 2]
        return raw[0::2].astype(np.float64) + \
            1j * raw[1::2].astype(np.float64)
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw[: raw.size // 2 * 2]
    return (raw[0::2].astype(np.float64) - 127.5) / 127.4 + \
        1j * ((raw[1::2].astype(np.float64) - 127.5) / 127.4)


# ---------------------------------------------------------------------
# PNG, written and read with the standard library only. Grayscale
# 8-bit, filter 0 on write; the reader understands all five filter
# types so it can read its own files and simple foreign ones.
# ---------------------------------------------------------------------

def write_png(path, image):
    arr = np.asarray(image, dtype=np.uint8)
    height, width = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(height))

    def chunk(kind, data):
        out = struct.pack(">I", len(data)) + kind + data
        out += struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        return out

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)
    return path


def read_png(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    width = height = None
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            if depth != 8 or color != 0:
                raise ValueError("only 8-bit grayscale PNG is supported")
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width + 1
    out = np.zeros((height, width), dtype=np.uint8)
    prev = np.zeros(width, dtype=np.int16)
    for y in range(height):
        row = np.frombuffer(raw[y * stride + 1:(y + 1) * stride],
                            dtype=np.uint8).astype(np.int16)
        ftype = raw[y * stride]
        if ftype == 0:
            cur = row
        elif ftype == 1:
            cur = row.copy()
            for x in range(1, width):
                cur[x] = (cur[x] + cur[x - 1]) & 0xFF
        elif ftype == 2:
            cur = (row + prev) & 0xFF
        elif ftype == 3:
            cur = row.copy()
            for x in range(width):
                left = cur[x - 1] if x else 0
                cur[x] = (cur[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:
            cur = row.copy()
            for x in range(width):
                a = cur[x - 1] if x else 0
                b = prev[x]
                c = prev[x - 1] if x else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else \
                    (b if pb <= pc else c)
                cur[x] = (cur[x] + pred) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter {ftype}")
        out[y] = cur.astype(np.uint8)
        prev = cur
    return out


# ---------------------------------------------------------------------
# The built-in test pattern: high contrast, named shapes, large block
# letters. Drawn on the visible raster only.
# ---------------------------------------------------------------------

_FONT = {
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
}


def make_pattern(width, height, text="VAN"):
    img = np.zeros((height, width), dtype=np.float64)
    # Border frame, three pixels in from the edge.
    b = 3
    img[b:b + 3, b:width - b] = 1.0
    img[height - b - 3:height - b, b:width - b] = 1.0
    img[b:height - b, b:b + 3] = 1.0
    img[b:height - b, width - b - 3:width - b] = 1.0
    # Block letters, centered in the upper two thirds.
    scale = max(2, (height * 2 // 3) // 9)
    glyph_w, glyph_h = 5 * scale, 7 * scale
    gap = 2 * scale
    total_w = len(text) * glyph_w + (len(text) - 1) * gap
    x0 = (width - total_w) // 2
    y0 = (height - glyph_h) // 3
    for li, ch in enumerate(text):
        glyph = _FONT[ch]
        gx = x0 + li * (glyph_w + gap)
        for row in range(7):
            for col in range(5):
                if glyph[row] & (1 << (4 - col)):
                    img[y0 + row * scale:y0 + (row + 1) * scale,
                        gx + col * scale:gx + (col + 1) * scale] = 1.0
    # A filled block in the lower right and a checker in the lower
    # left, so position within the frame is identifiable at a glance.
    bw, bh = width // 6, height // 6
    img[height - bh - 10:height - 10, width - bw - 10:width - 10] = 1.0
    cell = max(4, width // 40)
    for cy in range(4):
        for cx in range(4):
            if (cx + cy) % 2 == 0:
                y = height - 10 - 4 * cell + cy * cell
                x = 10 + cx * cell
                if 0 <= y and y + cell <= height and x + cell <= width:
                    img[y:y + cell, x:x + cell] = 1.0
    return img


def simulate_iq(mode, seconds=0.3, rate=None, snr_db=12.0,
                clock_ppm=37.0, seed=20261111, text="VAN"):
    """Synthesize a raster emanation.

    The envelope is the raster signal itself: pattern brightness in
    the visible slot range, a low baseline in blanking. It amplitude-
    modulates a carrier at baseband and takes complex noise. The
    effective pixel clock is offset by clock_ppm parts per million
    from the stated one, so sync estimation has something real to
    find. Returns (iq, info) where info is the ground truth.
    """
    rng = np.random.default_rng(seed)
    width, height = mode["width"], mode["height"]
    htotal, vtotal = mode["htotal"], mode["vtotal"]
    pattern = make_pattern(width, height, text=text)
    pclk_true = mode["pixel_clock"] * (1.0 + clock_ppm / 1e6)
    if rate is None:
        rate = 2 * mode["pixel_clock"]
    spp_true = rate / pclk_true
    slots_per_frame = htotal * vtotal
    n_frames = max(2, int(seconds * mode["refresh"]))
    total_slots = slots_per_frame * n_frames
    slot_values = np.full(total_slots, 0.08, dtype=np.float64)
    frame_slots = np.arange(slots_per_frame)
    line = frame_slots // htotal
    col = frame_slots % htotal
    visible = (line < height) & (col < width)
    one_frame = np.full(slots_per_frame, 0.08, dtype=np.float64)
    one_frame[visible] = pattern[line[visible], col[visible]]
    slot_values = np.tile(one_frame, n_frames)
    n_samples = int(total_slots * spp_true)
    sample_slot = np.minimum(
        (np.arange(n_samples) / spp_true).astype(np.int64),
        total_slots - 1)
    envelope = slot_values[sample_slot]
    t = np.arange(n_samples) / rate
    iq = envelope * np.exp(2j * np.pi * 0.0 * t)
    p_sig = float(np.mean(envelope ** 2))
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    sigma = math.sqrt(p_noise / 2.0)
    noise = sigma * (rng.standard_normal(n_samples)
                     + 1j * rng.standard_normal(n_samples))
    iq = iq + noise
    info = {
        "mode": mode["name"], "width": width, "height": height,
        "htotal": htotal, "vtotal": vtotal,
        "pixel_clock_stated": mode["pixel_clock"],
        "pixel_clock_true": pclk_true,
        "refresh": mode["refresh"], "rate": rate,
        "seconds": n_frames / mode["refresh"], "frames": n_frames,
        "snr_db": snr_db, "clock_ppm": clock_ppm,
        "samples": int(n_samples), "pattern_text": text,
    }
    return iq.astype(np.complex128), info, pattern


# ---------------------------------------------------------------------
# The bench. Reconstruction is folding: estimate the true frame
# period by autocorrelation, find the phase that makes the folded
# frames agree most sharply, then average every frame onto one
# raster. Every parameter searched is reported by the caller.
# ---------------------------------------------------------------------

def _period_autocorr(envelope, rate, refresh_nominal):
    """The envelope's autocorrelation, decimated as the period
    estimator decimates, as (ac, decim)."""
    nominal = rate / refresh_nominal
    decim = max(1, int(nominal // 150_000))
    if decim > 1:
        n = (len(envelope) // decim) * decim
        env = envelope[:n].reshape(-1, decim).mean(axis=1)
    else:
        env = envelope.astype(np.float64)
    env = env - env.mean()
    n = len(env)
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(env, nfft)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:n]
    return ac, decim


def estimate_frame_period(envelope, rate, refresh_nominal,
                          search_frac=0.03):
    """Autocorrelation of the envelope around the nominal frame lag.

    Returns (period_samples, peak_strength): the refined period as a
    float, and the autocorrelation peak normalized so that a perfect
    periodic signal scores near 1.
    """
    nominal = rate / refresh_nominal
    ac, decim = _period_autocorr(envelope, rate, refresh_nominal)
    n = len(ac)
    if ac[0] <= 0:
        return nominal, 0.0
    nom_d = nominal / decim
    lo = max(1, int(nom_d * (1.0 - search_frac)))
    hi = min(n - 1, int(nom_d * (1.0 + search_frac)) + 1)
    window = ac[lo:hi]
    if window.size == 0:
        return nominal, 0.0
    k = int(np.argmax(window)) + lo
    strength = float(ac[k] / ac[0])
    # Parabolic refinement around the peak, in decimated samples.
    if 0 < k < n - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    period = (k + delta) * decim
    return float(period), strength


def refine_period_multilag(envelope, rate, refresh_nominal, period0,
                           max_mult=6):
    """Refine a frame-period estimate against the autocorrelation
    peaks at several frame multiples.

    The peak at m frames sits at m * period, so its position error
    divided by m is a period estimate with m times less
    interpolation error; the single-lag estimate carries a small
    systematic bias (measured at about +0.25 ppm on the simulator's
    signal) that smears long folds sample by sample. Peaks are
    weighted by height and by the overlap supporting them, and a
    peak found at the edge of its narrow search window is
    distrusted and skipped. Returns period0 unchanged when no
    usable peak is found.
    """
    ac, decim = _period_autocorr(envelope, rate, refresh_nominal)
    if ac[0] <= 0:
        return period0
    n = len(ac)
    p0d = period0 / decim
    estimates, weights = [], []
    for m in range(1, max_mult + 1):
        center = m * p0d
        if center > n - 2:
            break
        halfwin = max(2.0, 0.0003 * center)
        lo = max(1, int(center - halfwin))
        hi = min(n - 1, int(center + halfwin) + 1)
        if hi - lo < 3:
            continue
        k = int(np.argmax(ac[lo:hi])) + lo
        if k == lo or k == hi - 1 or ac[k] <= 0:
            continue
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        estimates.append((k + delta) / m)
        weights.append(float(ac[k]) * (n - k))
    if not estimates:
        return period0
    w = np.asarray(weights)
    e = np.asarray(estimates)
    return float((w * e).sum() / w.sum()) * decim


def _fold_slots(envelope, period, phase, samples_per_slot, slot_count,
                max_frames=None):
    """Fold the envelope onto slot_count slots per frame.

    Returns (mean, var, frames): per-slot mean over frames, per-slot
    variance across frames, and the number of whole frames folded.
    """
    frames_avail = int((len(envelope) - phase) // period)
    frames = frames_avail if max_frames is None else \
        min(frames_avail, max_frames)
    if frames < 1:
        return None, None, 0
    slot_offsets = (np.arange(slot_count) + 0.5) * samples_per_slot
    total = np.zeros(slot_count, dtype=np.float64)
    total_sq = np.zeros(slot_count, dtype=np.float64)
    for f in range(frames):
        positions = phase + f * period + slot_offsets
        values = np.interp(positions, np.arange(len(envelope)),
                           envelope)
        total += values
        total_sq += values * values
    mean = total / frames
    var = np.maximum(total_sq / frames - mean * mean, 0.0)
    return mean, var, frames


def _phase_search(envelope, period, samples_per_slot, slot_count,
                  htotal, width, height):
    """Find the frame phase whose folded image has the most variance.

    Only visible slots are folded, on a strided subset and a few
    frames, so the search stays cheap at any mode size. Misalignment
    mixes blanking into the picture and flattens it; the true phase
    is the sharp one.
    """
    line_idx = np.arange(slot_count) // htotal
    col_idx = np.arange(slot_count) % htotal
    visible = np.nonzero((line_idx < height) & (col_idx < width))[0]
    stride = max(1, len(visible) // 60_000)
    subset = visible[::stride]
    slot_offsets = (subset + 0.5) * samples_per_slot
    frames_avail = int(len(envelope) // period)
    frames = max(1, min(frames_avail, 6))
    xp = np.arange(len(envelope))

    def score(phase):
        acc = np.zeros(len(subset), dtype=np.float64)
        for f in range(frames):
            acc += np.interp(phase + f * period + slot_offsets, xp,
                             envelope)
        return float(np.var(acc / frames))

    coarse_steps = 96
    step = period / coarse_steps
    best_phase, best_score = 0.0, -1.0
    for i in range(coarse_steps):
        phase = i * step
        s = score(phase)
        if s > best_score:
            best_phase, best_score = phase, s
    for frac in np.linspace(-1.0, 1.0, 17):
        phase = (best_phase + frac * step) % period
        s = score(phase)
        if s > best_score:
            best_phase, best_score = phase, s
    return best_phase, best_score


def auto_contrast(image, lo_pct=1.0, hi_pct=99.0):
    lo = float(np.percentile(image, lo_pct))
    hi = float(np.percentile(image, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (image - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalized_cross_correlation(a, b):
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("shape mismatch for correlation")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def reconstruct_iq(iq, rate, mode_params, reference=None):
    """Run the bench on loaded IQ.

    mode_params carries width, height, htotal, vtotal, refresh and
    (optionally) pixel_clock. Returns (image_uint8, report) where
    report holds every measured number.
    """
    width = mode_params["width"]
    height = mode_params["height"]
    htotal = mode_params["htotal"]
    vtotal = mode_params["vtotal"]
    refresh = mode_params["refresh"]
    envelope = np.abs(iq)
    period, ac_strength = estimate_frame_period(envelope, rate, refresh)
    period = refine_period_multilag(envelope, rate, refresh, period)
    slot_count = htotal * vtotal
    samples_per_slot = period / slot_count
    phase, phase_score = _phase_search(
        envelope, period, samples_per_slot, slot_count,
        htotal, width, height)
    mean, var, frames = _fold_slots(
        envelope, period, phase, samples_per_slot, slot_count)
    raster = mean.reshape(vtotal, htotal)
    visible = raster[:height, :width]
    image = auto_contrast(visible)
    # Image SNR: per-slot variance across frames, divided by the
    # frame count, is the noise left in the average.
    noise_var = float(np.mean(var)) / max(frames, 1)
    signal_var = float(np.var(mean))
    signal_only = max(signal_var - noise_var, 0.0)
    snr_db = 10.0 * math.log10(signal_only / noise_var) \
        if noise_var > 0 and signal_only > 0 else 0.0
    report = {
        "samples": int(len(iq)),
        "rate": rate,
        "refresh_nominal": refresh,
        "frame_period_samples": period,
        "frame_period_error_ppm":
            (period - rate / refresh) / (rate / refresh) * 1e6,
        "autocorr_peak": ac_strength,
        "phase_samples": phase,
        "phase_score": phase_score,
        "samples_per_slot": samples_per_slot,
        "frames_averaged": int(frames),
        "image_snr_db": snr_db,
        "width": width, "height": height,
        "htotal": htotal, "vtotal": vtotal,
    }
    if mode_params.get("pixel_clock"):
        report["pixel_clock_nominal"] = mode_params["pixel_clock"]
    if reference is not None:
        report["reference_ncc"] = normalized_cross_correlation(
            reference, image.astype(np.float64))
    return image, report


# ---------------------------------------------------------------------
# The live bench. watch folds frames as the signal arrives instead
# of waiting for a finished capture. Sync is acquired from the
# opening samples with the same estimator the offline bench uses,
# frames are accumulated with the same slot geometry, and the
# period estimate is re-checked against the recent signal on a
# cadence, so a drifting clock is tracked - corrections and
# re-locks are reported, never hidden. A frame here is the offline
# bench's frame: period from estimate_frame_period, phase from
# _phase_search, slot centers at (k + 0.5) * period / slot_count.
# ---------------------------------------------------------------------

class LiveFolder:
    """Incremental frame folder over a stream of envelope samples.

    Feed envelope chunks with push(); read the current picture with
    snapshot(). Frames already folded stay folded - a period
    adopted at a re-lock steers the frames that come after it, it
    does not rewrite the average that came before.
    """

    def __init__(self, params, rate, sync_frames=3.2, relock_every=30,
                 relock_tol=1e-4):
        self.width = params["width"]
        self.height = params["height"]
        self.htotal = params["htotal"]
        self.vtotal = params["vtotal"]
        self.refresh = params["refresh"]
        self.rate = rate
        self.slot_count = self.htotal * self.vtotal
        self.nominal_period = rate / self.refresh
        self.sync_min = int(sync_frames * self.nominal_period)
        self.relock_every = relock_every
        self.relock_tol = relock_tol
        # Stream state. buf holds every unconsumed sample, from
        # absolute index buf_start up to samples_seen.
        self.samples_seen = 0
        self.buf = np.zeros(0, dtype=np.float64)
        self.buf_start = 0
        # A short recent history, kept for the drift re-checks.
        self.ring = np.zeros(0, dtype=np.float64)
        self.ring_cap = int(7 * self.nominal_period) + 4096
        # Sync state.
        self.locked = False
        self.period = None
        self.sps = None
        self.next_start = 0.0   # absolute index of the next frame
        self.ac_strength = 0.0
        self.period_checks = []  # every re-check, as measured
        self.relocks = 0
        self.corrections = 0
        self.frames_at_last_check = 0
        # Folded moments: per-slot sum and sum of squares.
        self.sum_slots = np.zeros(self.slot_count, dtype=np.float64)
        self.sumsq_slots = np.zeros(self.slot_count, dtype=np.float64)
        self.frames_folded = 0

    # -- intake ----------------------------------------------------

    def push(self, envelope_chunk):
        chunk = np.asarray(envelope_chunk, dtype=np.float64)
        if chunk.size == 0:
            return
        self.samples_seen += int(chunk.size)
        self.buf = np.concatenate([self.buf, chunk]) \
            if self.buf.size else chunk.copy()
        self.ring = np.concatenate([self.ring, chunk]) \
            if self.ring.size else chunk.copy()
        if len(self.ring) > self.ring_cap:
            self.ring = self.ring[-self.ring_cap:]
        if not self.locked:
            if len(self.buf) >= self.sync_min:
                self._acquire()
            return
        self._fold_available()
        self._maybe_relock()

    def flush(self):
        """End of stream: acquire sync with what arrived, if it is
        enough, and fold whatever whole frames remain."""
        if not self.locked and len(self.buf) >= \
                int(2.5 * self.nominal_period):
            self._acquire()
        if self.locked:
            self._fold_available()

    # -- sync --------------------------------------------------------

    def _acquire(self):
        period, strength = estimate_frame_period(
            self.buf, self.rate, self.refresh)
        period = refine_period_multilag(
            self.buf, self.rate, self.refresh, period)
        sps = period / self.slot_count
        phase, _score = _phase_search(
            self.buf, period, sps, self.slot_count,
            self.htotal, self.width, self.height)
        self.period = float(period)
        self.sps = float(sps)
        self.ac_strength = float(strength)
        self.next_start = float(phase)
        self.locked = True
        self.frames_at_last_check = 0
        self._fold_available()

    def _maybe_relock(self):
        if self.frames_folded - self.frames_at_last_check \
                < self.relock_every:
            return
        if len(self.ring) < int(3 * self.nominal_period):
            return
        self.frames_at_last_check = self.frames_folded
        raw_period, strength = estimate_frame_period(
            self.ring, self.rate, self.refresh)
        period = refine_period_multilag(
            self.ring, self.rate, self.refresh, raw_period)
        check = {"at_frame": self.frames_folded,
                 "period_samples": float(period),
                 "period_delta_ppm":
                     (period - self.period) / self.period * 1e6,
                 "autocorr_peak": float(strength),
                 "adopted": False, "corrected": False}
        self.period_checks.append(check)
        if strength < 0.1:
            return  # too weak to trust; keep the period we have
        delta = abs(period - self.period) / self.period
        if delta <= 5e-7:
            return  # the clock agrees with the tracked period
        if delta <= self.relock_tol:
            # A correction, not a re-lock: the frame grid stands,
            # only the spacing of the frames that follow changes.
            self.period = float(period)
            self.sps = period / self.slot_count
            self.ac_strength = float(strength)
            check["corrected"] = True
            self.corrections += 1
            return
        # Adopt: re-find the phase inside the ring, then place the
        # next frame start on the new grid, at or after the old one,
        # so no sample is folded twice and less than one frame of
        # signal is ever skipped.
        sps = period / self.slot_count
        phase, _score = _phase_search(
            self.ring, period, sps, self.slot_count,
            self.htotal, self.width, self.height)
        ring_start = self.samples_seen - len(self.ring)
        first = ring_start + phase
        k = math.ceil((self.next_start - first) / period - 1e-9)
        self.period = float(period)
        self.sps = float(sps)
        self.next_start = float(first + k * period)
        self.ac_strength = float(strength)
        check["adopted"] = True
        self.relocks += 1

    # -- folding -----------------------------------------------------

    def _fold_available(self):
        while self.period is not None:
            frames_avail = int((self.samples_seen - self.next_start)
                               // self.period)
            if frames_avail < 1:
                return
            k = min(frames_avail, 16)
            rel_start = self.next_start - self.buf_start
            slot_offsets = (np.arange(self.slot_count) + 0.5) \
                * self.sps
            starts = rel_start + np.arange(k) * self.period
            positions = (starts[:, None]
                         + slot_offsets[None, :]).ravel()
            xp = np.arange(len(self.buf))
            vals = np.interp(positions, xp, self.buf)
            vals = vals.reshape(k, self.slot_count)
            self.sum_slots += vals.sum(axis=0)
            self.sumsq_slots += (vals * vals).sum(axis=0)
            self.frames_folded += k
            self.next_start += k * self.period
            new_start = int(self.next_start)
            if new_start > self.buf_start:
                drop = min(new_start - self.buf_start, len(self.buf))
                self.buf = self.buf[drop:]
                self.buf_start += drop

    # -- the picture so far ------------------------------------------

    def snapshot(self):
        """(image, stats) for the frames folded so far, or
        (None, stats) before the first frame lands. The image SNR
        is the offline bench's computation on the running moments:
        per-slot variance across frames, divided by the frame
        count, is the noise left in the average."""
        stats = {
            "locked": self.locked,
            "frames_folded": self.frames_folded,
            "samples_seen": self.samples_seen,
            "frame_period_samples": self.period,
            "frame_period_error_ppm":
                (self.period - self.nominal_period)
                / self.nominal_period * 1e6 if self.period else None,
            "autocorr_peak": self.ac_strength,
            "relocks": self.relocks,
            "corrections": self.corrections,
            "period_checks": len(self.period_checks),
            "image_snr_db": 0.0,
        }
        if self.frames_folded < 1:
            return None, stats
        mean = self.sum_slots / self.frames_folded
        var = np.maximum(
            self.sumsq_slots / self.frames_folded - mean * mean, 0.0)
        raster = mean.reshape(self.vtotal, self.htotal)
        image = auto_contrast(raster[:self.height, :self.width])
        noise_var = float(np.mean(var)) / self.frames_folded
        signal_var = float(np.var(mean))
        signal_only = max(signal_var - noise_var, 0.0)
        if noise_var > 0 and signal_only > 0:
            stats["image_snr_db"] = 10.0 * math.log10(
                signal_only / noise_var)
        return image, stats


# ---------------------------------------------------------------------
# Stream sources. A source yields (raw_bytes, iq) blocks: the raw
# bytes exactly as they arrived, so watch can record the stream
# byte-for-byte while it folds, and the decoded IQ with load_iq's
# scaling. Two sources: a file replayed as a stream, and a live
# rtl_sdr process writing unsigned 8-bit IQ to its stdout.
# ---------------------------------------------------------------------

def _decode_block(raw, fmt):
    if fmt == "cf32":
        arr = np.frombuffer(raw, dtype=np.float32)
        arr = arr[: arr.size // 2 * 2]
        return arr[0::2].astype(np.float64) \
            + 1j * arr[1::2].astype(np.float64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    arr = arr[: arr.size // 2 * 2]
    return (arr[0::2].astype(np.float64) - 127.5) / 127.4 \
        + 1j * ((arr[1::2].astype(np.float64) - 127.5) / 127.4)


def replay_chunks(path, chunk_samples):
    """Replay an IQ file (.cu8 or .cf32) as a stream of blocks."""
    path = Path(path)
    fmt = "cf32" if path.suffix.lower() in (".cf32", ".iq") else "cu8"
    bytes_per_sample = 8 if fmt == "cf32" else 2
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(chunk_samples * bytes_per_sample)
            if not raw:
                return
            yield raw, _decode_block(raw, fmt)


def rtl_sdr_chunks(plan, chunk_samples):
    """Stream from a live rtl_sdr process. The generator owns the
    process: closing the generator terminates it."""
    cmd = ["rtl_sdr", "-f", str(plan["freq_hz"]), "-s",
           str(plan["rate"])]
    if plan.get("gain") is not None:
        cmd += ["-g", str(plan["gain"])]
    cmd += ["-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = proc.stdout.read(chunk_samples * 2)
            if not raw:
                return
            yield raw, _decode_block(raw, "cu8")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


# ---------------------------------------------------------------------
# Target selection. capture and reconstruct take a target by number
# from the last hunt, or a manual frequency / explicit mode. With a
# terminal attached and nothing specified, the operator is shown
# the list and asked.
# ---------------------------------------------------------------------

def load_target(session, number):
    payload = session.load_targets()
    if not payload:
        return None
    for t in payload.get("targets", []):
        if t.get("number") == number:
            return t
    return None


def prompt_for_target(session, payload):
    print("targets from the last hunt:")
    for line in format_target_list(payload.get("targets", [])):
        print(line)
    while True:
        try:
            answer = input("select target number (blank to cancel): ")
        except EOFError:
            return None
        if not answer.strip():
            return None
        if answer.strip().isdigit():
            target = load_target(session, int(answer.strip()))
            if target is not None:
                return target
        print("no such target; try again")


def target_mode_params(target):
    if target is None or target.get("mode_guess") is None:
        return None
    return {"width": target["width"], "height": target["height"],
            "htotal": target["htotal"], "vtotal": target["vtotal"],
            "pixel_clock": target["pixel_clock"],
            "refresh": target["refresh_est"]}


def plan_capture(target=None, freq=None, rate=2_400_000, seconds=5.0,
                 gain=None):
    """The capture parameters a selection locks in, before any
    hardware is touched."""
    if target is not None:
        freq = target["strongest_freq_hz"]
    return {
        "freq_hz": int(freq) if freq is not None else None,
        "rate": int(rate), "seconds": float(seconds),
        "samples": int(rate * seconds), "gain": gain,
        "target_number": target["number"] if target else None,
        "target_mode": target["mode_guess"] if target else None,
    }


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

def _average_spectrum(freqs, db):
    merged = {}
    for f, v in zip(freqs, db):
        merged.setdefault(f, []).append(v)
    keys = sorted(merged)
    return keys, [sum(merged[k]) / len(merged[k]) for k in keys]


def command_hunt(args) -> int:
    session = Session(args.session)
    hunt_info = {"low_hz": parse_freq(args.low),
                 "high_hz": parse_freq(args.high),
                 "threshold_db": args.threshold}
    if args.spectrum:
        freqs, db = load_spectrum_file(args.spectrum)
        hunt_info["source"] = f"file:{args.spectrum}"
    else:
        holders = refuse_if_busy(session, "hunt")
        if holders:
            print_busy(holders)
            return EXIT_BUSY
        if shutil.which("rtl_power") is None:
            print("hunt unavailable: rtl_power not found, and no "
                  "--spectrum file was given", file=sys.stderr)
            session.log("hunt_unavailable", reason="rtl_power missing")
            return EXIT_UNAVAILABLE
        csv_path = session.root / "hunt-spectrum.csv"
        cmd = ["rtl_power", "-f",
               f"{hunt_info['low_hz']}:{hunt_info['high_hz']}:"
               f"{args.bin_hz}",
               "-e", f"{args.seconds}s"]
        if args.gain is not None:
            cmd += ["-g", str(args.gain)]
        cmd += [str(csv_path)]
        session.log("hunt_started", **hunt_info, command=cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True)
        except subprocess.CalledProcessError as exc:
            print(f"rtl_power failed: {exc.stderr.strip()}",
                  file=sys.stderr)
            session.log("hunt_failed", error=exc.stderr.strip())
            return EXIT_UNAVAILABLE
        freqs, db = load_spectrum_file(csv_path)
        hunt_info["source"] = "rtl_power"
    if not freqs:
        print("hunt found no spectrum data to read", file=sys.stderr)
        session.log("hunt_empty", **hunt_info)
        return EXIT_UNAVAILABLE
    freqs, db = _average_spectrum(freqs, db)
    targets = find_targets(freqs, db, threshold_db=args.threshold)
    session.save_targets(targets, hunt_info)
    session.log("hunt", **hunt_info, bins=len(freqs),
                targets=len(targets))
    print(f"hunt: {len(freqs)} bins, "
          f"{hunt_info['low_hz'] / 1e6:.1f}-"
          f"{hunt_info['high_hz'] / 1e6:.1f} MHz, "
          f"floor-relative threshold {args.threshold:.1f} dB")
    for line in format_target_list(targets):
        print(line)
    print(f"target list saved to {session.targets_path}")
    return EXIT_OK


def _resolve_target(args, session, allow_prompt=True):
    """Returns (target, error_code). target may be None when the
    caller supplied a manual frequency instead."""
    if getattr(args, "target", None) is not None:
        target = load_target(session, args.target)
        if target is None:
            print(f"no target {args.target} in the last hunt "
                  f"({session.targets_path})", file=sys.stderr)
            return None, EXIT_UNAVAILABLE
        return target, None
    if allow_prompt and getattr(args, "freq", None) is None \
            and sys.stdin.isatty():
        payload = session.load_targets()
        if payload and payload.get("targets"):
            target = prompt_for_target(session, payload)
            if target is None:
                print("no target selected", file=sys.stderr)
                return None, EXIT_UNAVAILABLE
            return target, None
    return None, None


def command_capture(args) -> int:
    session = Session(args.session)
    target, err = _resolve_target(args, session)
    if err is not None:
        return err
    if target is None and args.freq is None:
        print("capture needs a target (--target N) or a frequency "
              "(--freq); run hunt first", file=sys.stderr)
        return EXIT_UNAVAILABLE
    plan = plan_capture(target=target,
                        freq=parse_freq(args.freq) if args.freq
                        else None,
                        rate=args.rate, seconds=args.seconds,
                        gain=args.gain)
    # The plan, including the selected target, is written down
    # before the tuner is touched.
    session.log("capture_planned", **plan)
    holders = refuse_if_busy(session, "capture")
    if holders:
        print_busy(holders)
        return EXIT_BUSY
    if shutil.which("rtl_sdr") is None:
        print("capture unavailable: rtl_sdr not found",
              file=sys.stderr)
        session.log("capture_unavailable", reason="rtl_sdr missing",
                    **plan)
        return EXIT_UNAVAILABLE
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else \
        session.root / f"capture-{stamp}.cu8"
    cmd = ["rtl_sdr", "-f", str(plan["freq_hz"]), "-s",
           str(plan["rate"]), "-n", str(plan["samples"])]
    if plan["gain"] is not None:
        cmd += ["-g", str(plan["gain"])]
    cmd += [str(out)]
    print(f"capture: {fmt_freq(plan['freq_hz'])}, "
          f"{plan['rate']} samples/s, {plan['seconds']} s"
          + (f", target {plan['target_number']} "
             f"({plan['target_mode']})" if target else "")
          + f" -> {out}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"rtl_sdr failed: {exc.stderr.strip()}",
              file=sys.stderr)
        session.log("capture_failed", error=exc.stderr.strip(), **plan)
        return EXIT_UNAVAILABLE
    size = out.stat().st_size if out.is_file() else 0
    session.log("capture_done", file=str(out), bytes=size, **plan)
    print(f"captured {size} bytes")
    return EXIT_OK


def command_watch(args) -> int:
    if not HAS_NUMPY:
        print("watch unavailable: numpy is not installed",
              file=sys.stderr)
        return EXIT_UNAVAILABLE
    session = Session(args.session)
    target, err = _resolve_target(args, session)
    if err is not None:
        return err
    source_path = Path(args.source) if args.source else None
    if source_path is None and target is None and args.freq is None:
        print("watch needs a target (--target N), a frequency "
              "(--freq), or a replay file (--source); run hunt first",
              file=sys.stderr)
        return EXIT_UNAVAILABLE
    params = _resolve_mode_params(args, target)
    if "width" not in params or "height" not in params:
        print("watch needs a mode: --target N, --mode NAME, or "
              "--width and --height", file=sys.stderr)
        return EXIT_UNAVAILABLE
    plan = {
        "freq_hz": int(target["strongest_freq_hz"]) if target
        else (parse_freq(args.freq) if args.freq else None),
        "rate": int(args.rate),
        "gain": args.gain,
        "target_number": target["number"] if target else None,
        "target_mode": target["mode_guess"] if target else None,
        "source": str(source_path) if source_path else "rtl_sdr",
        "mode": f"{params['width']}x{params['height']}",
        "record_enabled": not args.no_record,
    }
    # The plan is written down before the tuner is touched, as with
    # capture.
    session.log("watch_planned", **plan)
    chunk_samples = max(1024, int(args.rate * args.chunk_seconds))
    if source_path is not None:
        if not source_path.is_file():
            print(f"no such IQ file: {source_path}",
                  file=sys.stderr)
            session.log("watch_unavailable",
                        reason="source missing", **plan)
            return EXIT_UNAVAILABLE
        stream = replay_chunks(source_path, chunk_samples)
        record_ext = source_path.suffix.lower() \
            if source_path.suffix.lower() in (".cu8", ".cf32") \
            else ".cf32"
    else:
        holders = refuse_if_busy(session, "watch")
        if holders:
            print_busy(holders)
            return EXIT_BUSY
        if shutil.which("rtl_sdr") is None:
            print("watch unavailable: rtl_sdr not found",
                  file=sys.stderr)
            session.log("watch_unavailable",
                        reason="rtl_sdr missing", **plan)
            return EXIT_UNAVAILABLE
        stream = rtl_sdr_chunks(plan, chunk_samples)
        record_ext = ".cu8"

    reference = None
    if args.reference:
        reference = read_png(args.reference).astype(np.float64)
    out = Path(args.out) if args.out else session.root / "watch.png"
    first_out = out.with_name(out.stem + "-first.png")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record_path = None
    if not args.no_record:
        record_path = Path(args.record_out) if args.record_out \
            else session.root / f"watch-{stamp}{record_ext}"
    folder = LiveFolder(params, args.rate)
    print(f"watch: {plan['mode']}, rate {args.rate} samples/s, "
          + (f"replaying {source_path}" if source_path else
             f"{fmt_freq(plan['freq_hz'])}"
             + (f", target {plan['target_number']} "
                f"({plan['target_mode']})" if target else ""))
          + f" -> {out}")
    if record_path is not None:
        print(f"  recording raw stream to {record_path}")

    sink = None
    started = time.monotonic()
    sample_cap = int(args.seconds * args.rate) \
        if args.seconds else None
    last_write_wall = 0.0
    last_write_frames = 0
    wrote_first = False
    announced_lock = False

    def write_snapshot(final=False):
        nonlocal last_write_wall, last_write_frames, wrote_first
        image, stats = folder.snapshot()
        if image is None:
            return stats
        write_png(out, image)
        if not wrote_first:
            write_png(first_out, image)
            wrote_first = True
        line = (f"  {stats['frames_folded']} frames folded, "
                f"image SNR {stats['image_snr_db']:.1f} dB")
        if reference is not None:
            ncc = normalized_cross_correlation(
                reference, image.astype(np.float64))
            line += f", correlation {ncc:.3f}"
            stats["reference_ncc"] = ncc
        line += f" -> {out}" if final else ""
        print(line, flush=True)
        last_write_wall = time.monotonic()
        last_write_frames = stats["frames_folded"]
        return stats

    stats = folder.snapshot()[1]
    try:
        while True:
            try:
                raw, iq = next(stream)
            except StopIteration:
                break
            if sink is None and record_path is not None:
                sink = open(record_path, "wb")
            if sink is not None:
                sink.write(raw)
            relocks_before = folder.relocks
            corrections_before = folder.corrections
            folder.push(np.abs(iq))
            if folder.locked and not announced_lock:
                announced_lock = True
                session.log(
                    "watch_locked",
                    frame_period_samples=folder.period,
                    frame_period_error_ppm=
                    (folder.period - folder.nominal_period)
                    / folder.nominal_period * 1e6,
                    autocorr_peak=folder.ac_strength)
                print(f"  sync locked: period {folder.period:.1f} "
                      f"samples ({(folder.period - folder.nominal_period) / folder.nominal_period * 1e6:+.1f} "
                      f"ppm from nominal), autocorrelation peak "
                      f"{folder.ac_strength:.3f}", flush=True)
            if folder.relocks > relocks_before:
                check = folder.period_checks[-1]
                session.log("watch_relock", **check)
                print(f"  sync re-locked at frame "
                      f"{check['at_frame']}: period "
                      f"{folder.period:.1f} samples "
                      f"({check['period_delta_ppm']:+.1f} ppm "
                      f"correction), autocorrelation peak "
                      f"{check['autocorr_peak']:.3f}", flush=True)
            elif folder.corrections > corrections_before:
                check = folder.period_checks[-1]
                session.log("watch_correction", **check)
                print(f"  sync corrected at frame "
                      f"{check['at_frame']}: period "
                      f"{folder.period:.2f} samples "
                      f"({check['period_delta_ppm']:+.2f} ppm)",
                      flush=True)
            now = time.monotonic()
            if folder.frames_folded > last_write_frames and (
                    folder.frames_folded - last_write_frames
                    >= args.every_frames
                    or now - last_write_wall >= args.interval):
                stats = write_snapshot()
            if sample_cap is not None \
                    and folder.samples_seen >= sample_cap:
                break
    except KeyboardInterrupt:
        print("\nwatch interrupted", flush=True)
    finally:
        stream.close()
        if sink is not None:
            sink.close()
    folder.flush()
    stats = write_snapshot(final=True)
    wall = time.monotonic() - started
    record_bytes = record_path.stat().st_size \
        if record_path is not None and record_path.is_file() else 0
    summary = {
        "frames_folded": stats["frames_folded"],
        "samples": stats["samples_seen"],
        "signal_seconds": stats["samples_seen"] / args.rate,
        "wall_seconds": wall,
        "frame_period_samples": stats["frame_period_samples"],
        "frame_period_error_ppm": stats["frame_period_error_ppm"],
        "autocorr_peak": stats["autocorr_peak"],
        "relocks": stats["relocks"],
        "corrections": stats["corrections"],
        "period_checks": stats["period_checks"],
        "image_snr_db": stats["image_snr_db"],
        "image": str(out),
        "first_image": str(first_out) if wrote_first else None,
        "record": str(record_path) if record_path else None,
        "record_bytes": record_bytes,
    }
    if "reference_ncc" in stats:
        summary["reference_ncc"] = stats["reference_ncc"]
    session.log("watch", **plan, **summary)
    if stats["frames_folded"] < 1:
        print("watch: no sync acquired - no frame was folded, and "
              "no image formed")
    print(f"watch summary: {summary['frames_folded']} frames folded, "
          f"{summary['signal_seconds']:.1f} s of signal in "
          f"{wall:.1f} s")
    if summary["frame_period_samples"]:
        print(f"  period {summary['frame_period_samples']:.1f} "
              f"samples ({summary['frame_period_error_ppm']:+.1f} "
              f"ppm), sync peak {summary['autocorr_peak']:.3f}, "
              f"{summary['relocks']} re-locks and "
              f"{summary['corrections']} corrections over "
              f"{summary['period_checks']} checks")
    print(f"  image SNR {summary['image_snr_db']:.1f} dB")
    print(f"  image  {out}")
    if record_path is not None:
        print(f"  record {record_path} ({record_bytes} bytes)")
    return EXIT_OK


def _resolve_mode_params(args, target):
    params = {}
    if target is not None:
        from_target = target_mode_params(target)
        if from_target:
            params.update(from_target)
    if getattr(args, "mode", None):
        mode = MODE_BY_NAME[args.mode]
        params.update({"width": mode["width"],
                       "height": mode["height"],
                       "htotal": mode["htotal"],
                       "vtotal": mode["vtotal"],
                       "pixel_clock": mode["pixel_clock"],
                       "refresh": mode["refresh"]})
    for key in ("width", "height", "htotal", "vtotal", "refresh"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if getattr(args, "pixel_clock", None) is not None:
        params["pixel_clock"] = args.pixel_clock
    if "width" in params and "htotal" not in params:
        params["htotal"] = math.ceil(params["width"] * 1.25)
    if "height" in params and "vtotal" not in params:
        params["vtotal"] = math.ceil(params["height"] * 1.2)
    if "refresh" not in params:
        params["refresh"] = 60.0
    return params


def command_reconstruct(args) -> int:
    if not HAS_NUMPY:
        print("reconstruct unavailable: numpy is not installed",
              file=sys.stderr)
        return EXIT_UNAVAILABLE
    session = Session(args.session)
    target, err = _resolve_target(args, session, allow_prompt=False)
    if err is not None:
        return err
    params = _resolve_mode_params(args, target)
    if "width" not in params or "height" not in params:
        print("reconstruct needs a mode: --target N, --mode NAME, or "
              "--width and --height", file=sys.stderr)
        return EXIT_UNAVAILABLE
    iq_path = Path(args.iq)
    if not iq_path.is_file():
        print(f"no such IQ file: {iq_path}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    iq = load_iq(iq_path)
    reference = None
    if args.reference:
        reference = read_png(args.reference).astype(np.float64)
    image, report = reconstruct_iq(iq, args.rate, params,
                                   reference=reference)
    out = Path(args.out) if args.out else \
        session.root / (iq_path.stem + ".png")
    write_png(out, image)
    side_by_side = None
    if reference is not None:
        gap = np.zeros((image.shape[0], 8), dtype=np.uint8)
        combined = np.concatenate(
            [reference.astype(np.uint8), gap, image], axis=1)
        side_by_side = out.with_name(out.stem + "-side-by-side.png")
        write_png(side_by_side, combined)
    session.log("reconstruct", iq=str(iq_path), out=str(out),
                target_number=target["number"] if target else None,
                **report)
    print(f"reconstruct {iq_path}")
    print(f"  mode        {params['width']}x{params['height']} "
          f"(totals {params['htotal']}x{params['vtotal']}), "
          f"refresh nominal {params['refresh']} Hz")
    print(f"  searched    frame lag +/-3% of nominal "
          f"{args.rate / params['refresh']:.1f} samples, "
          f"phase over one full frame period")
    print(f"  period      {report['frame_period_samples']:.2f} "
          f"samples ({report['frame_period_error_ppm']:+.1f} ppm "
          f"from nominal)")
    print(f"  sync        autocorrelation peak "
          f"{report['autocorr_peak']:.3f}, phase "
          f"{report['phase_samples']:.1f} samples")
    print(f"  frames      {report['frames_averaged']} averaged, "
          f"{report['samples_per_slot']:.3f} samples per pixel slot")
    print(f"  image SNR   {report['image_snr_db']:.1f} dB")
    if "reference_ncc" in report:
        print(f"  reference   normalized cross-correlation "
              f"{report['reference_ncc']:.3f}")
    print(f"  image       {out}")
    if side_by_side:
        print(f"  side-by-side {side_by_side}")
    return EXIT_OK


def command_simulate(args) -> int:
    if not HAS_NUMPY:
        print("simulate unavailable: numpy is not installed",
              file=sys.stderr)
        return EXIT_UNAVAILABLE
    session = Session(args.session)
    mode = MODE_BY_NAME[args.mode]
    iq, info, pattern = simulate_iq(
        mode, seconds=args.seconds, rate=args.rate,
        snr_db=args.snr_db, clock_ppm=args.clock_ppm, seed=args.seed,
        text=args.text)
    out = Path(args.out) if args.out else session.root / "sim.cf32"
    save_iq(out, iq, "cf32")
    pattern_path = out.with_suffix(".pattern.png")
    write_png(pattern_path,
              (pattern * 255).astype(np.uint8))
    sidecar = out.with_suffix(".json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)
    session.log("simulate", out=str(out), **info)
    print(f"simulate: mode {info['mode']}, {info['frames']} frames, "
          f"{info['samples']} samples at {info['rate']} samples/s")
    print(f"  stated pixel clock {info['pixel_clock_stated']} Hz, "
          f"true {info['pixel_clock_true']:.1f} Hz "
          f"({info['clock_ppm']:+.0f} ppm), SNR {info['snr_db']} dB")
    print(f"  iq        {out}")
    print(f"  pattern   {pattern_path}")
    print(f"  ground truth {sidecar}")
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="catoptrics",
        description="A Van Eck emanation viewer: hunt emanation "
                    "sources, select a target, capture its signal, "
                    "and reconstruct the picture it is leaking.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--session", default="catoptrics-session",
                       help="session directory (default "
                            "./catoptrics-session)")

    hunt = sub.add_parser("hunt", help="census of emanation sources")
    common(hunt)
    hunt.add_argument("--low", default="40M")
    hunt.add_argument("--high", default="400M")
    hunt.add_argument("--bin-hz", type=int, default=25_000)
    hunt.add_argument("--seconds", type=int, default=20,
                      help="sweep time budget for rtl_power")
    hunt.add_argument("--gain", type=float, default=None)
    hunt.add_argument("--threshold", type=float, default=6.0,
                      help="dB over local floor a peak must reach")
    hunt.add_argument("--spectrum", default=None,
                      help="read a spectrum file instead of sweeping")
    hunt.set_defaults(func=command_hunt)

    capture = sub.add_parser("capture", help="record raw IQ")
    common(capture)
    capture.add_argument("--target", type=int, default=None)
    capture.add_argument("--freq", default=None)
    capture.add_argument("--rate", type=int, default=2_400_000)
    capture.add_argument("--seconds", type=float, default=5.0)
    capture.add_argument("--gain", type=float, default=None)
    capture.add_argument("--out", default=None)
    capture.set_defaults(func=command_capture)

    watch = sub.add_parser(
        "watch", help="fold a live stream as it arrives")
    common(watch)
    watch.add_argument("--target", type=int, default=None)
    watch.add_argument("--freq", default=None)
    watch.add_argument("--mode", choices=sorted(MODE_BY_NAME),
                       default=None)
    watch.add_argument("--width", type=int, default=None)
    watch.add_argument("--height", type=int, default=None)
    watch.add_argument("--htotal", type=int, default=None)
    watch.add_argument("--vtotal", type=int, default=None)
    watch.add_argument("--refresh", type=float, default=None)
    watch.add_argument("--pixel-clock", type=float, default=None)
    watch.add_argument("--rate", type=int, default=2_400_000)
    watch.add_argument("--gain", type=float, default=None)
    watch.add_argument("--source", default=None,
                       help="replay this IQ file (.cu8 or .cf32) "
                            "as a stream instead of using the tuner")
    watch.add_argument("--seconds", type=float, default=None,
                       help="stop after this many seconds of signal "
                            "(default: until the stream ends or "
                            "Ctrl+C)")
    watch.add_argument("--out", default=None,
                       help="live image path (default "
                            "<session>/watch.png), rewritten in place")
    watch.add_argument("--interval", type=float, default=1.0,
                       help="minimum seconds between snapshot writes")
    watch.add_argument("--every-frames", type=int, default=30,
                       help="also write a snapshot every N folded "
                            "frames")
    watch.add_argument("--chunk-seconds", type=float, default=0.25,
                       help="stream block size, in seconds of signal")
    watch.add_argument("--no-record", action="store_true",
                       help="do not record the raw stream while "
                            "watching")
    watch.add_argument("--record-out", default=None,
                       help="recording path (default "
                            "<session>/watch-<time>.<fmt>)")
    watch.add_argument("--reference", default=None,
                       help="reference PNG to score each snapshot "
                            "against")
    watch.set_defaults(func=command_watch)

    recon = sub.add_parser("reconstruct", help="fold IQ into an image")
    common(recon)
    recon.add_argument("--iq", required=True)
    recon.add_argument("--target", type=int, default=None)
    recon.add_argument("--mode", choices=sorted(MODE_BY_NAME),
                       default=None)
    recon.add_argument("--width", type=int, default=None)
    recon.add_argument("--height", type=int, default=None)
    recon.add_argument("--htotal", type=int, default=None)
    recon.add_argument("--vtotal", type=int, default=None)
    recon.add_argument("--refresh", type=float, default=None)
    recon.add_argument("--pixel-clock", type=float, default=None)
    recon.add_argument("--rate", type=int, default=2_400_000)
    recon.add_argument("--out", default=None)
    recon.add_argument("--reference", default=None,
                       help="reference PNG to score against")
    recon.set_defaults(func=command_reconstruct)

    sim = sub.add_parser("simulate", help="synthesize a test emanation")
    common(sim)
    sim.add_argument("--out", default=None)
    sim.add_argument("--mode", choices=sorted(MODE_BY_NAME),
                     default="SIM320x200@60")
    sim.add_argument("--seconds", type=float, default=0.3)
    sim.add_argument("--rate", type=int, default=None)
    sim.add_argument("--snr-db", type=float, default=12.0)
    sim.add_argument("--clock-ppm", type=float, default=37.0)
    sim.add_argument("--seed", type=int, default=20261111)
    sim.add_argument("--text", default="VAN")
    sim.set_defaults(func=command_simulate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
