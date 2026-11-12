# Catoptrics

A Van Eck emanation viewer. A computer display and its cable radiate
a faint radio copy of the image being shown; Catoptrics receives that
radiation with an RTL-SDR (the RTL2832U family) and reconstructs the
picture from it. It is receive-only: it transmits nothing, touches no
network, and reads a signal that is already in the air.

The name is for catoptrics, the branch of optics that treats mirrors
and reflected images. Every ordinary picture is made from light that
was aimed at the viewer; this one is made from a reflection nobody
aimed.

## Purpose

Catoptrics is a research instrument for measuring and viewing the
emanations of one's own displays, in one's own space. It exists to
answer, with numbers, a question usually argued from folklore: how
much of a screen's image actually leaves the room, and in what shape?

## The doctrine

- Every claim is a number the tool printed. Signal strength, sync
  quality, frames averaged, the correlation of a reconstruction
  against a known pattern - if a run cannot measure it, the run says
  so instead of estimating it.
- A null result is a result. A band with no emanation peak, a display
  that yields nothing reconstructable, is reported as exactly that,
  with the noise floor it was measured against.
- The tuner is shared and never seized. If another program already
  holds the receiver, Catoptrics refuses and names the holder; it
  does not kill, preempt, or queue behind anything.
- Nothing is faked. Reconstructed images come from the capture they
  are labelled with. The built-in simulator is a proving ground for
  the reconstruction bench, and its output is always labelled as
  simulated.

## How it is used

Hunt, read the list, select a target, then capture and reconstruct
- or watch the picture form live.

- `hunt` sweeps a band and takes a census. Peaks that belong to one
  source - one pixel-clock harmonic family with a consistent refresh
  cadence - are grouped into a single target. The result is a
  numbered target list, strongest first, persisted as `targets.json`
  in the session directory. Each entry shows its evidence: the mode
  it matches, its strongest frequency and measured level over the
  local floor, its estimated refresh, and whether a second harmonic
  confirms the family. A "target" is a distinct emanation source
  cluster as measured - no more than that claim - and weak or
  ambiguous clusters are marked as such, not presented as certain.
  A peak that two families could claim (the second harmonic of
  1280x720 is the fundamental of 1920x1080) is counted once, and the
  sharing is printed as a note on the target that claimed it.
- `capture` records raw IQ from the receiver. It takes a target by
  number (`--target N`, from the last hunt) or asks for one: run in
  a terminal with no target named, it prints the list and reads the
  operator's choice. Selecting a target locks the capture frequency
  and the expected mode - pixel clock, resolution, refresh - to that
  target's measured parameters, and the session log records which
  target was selected. The capture plan is written to the session
  log before the tuner is touched.
- `reconstruct` is the bench: envelope detection, frame-sync
  refinement by autocorrelation, folding into a raster, averaging
  across frames, and a written report of the parameters searched,
  the sync quality found, and the image SNR. It accepts the same
  `--target N`, which supplies the mode parameters the target was
  measured with; explicit flags override them.
- `simulate` synthesizes a raster emanation from a built-in test
  pattern, so the bench can be proven end to end without hardware.
- `watch` is the live mode: capture and reconstruction at the same
  time. It streams IQ from the receiver and folds frames as they
  arrive, so the picture forms while the signal is still coming in.
  Sync is acquired from the opening seconds of the stream with the
  same estimator the offline bench uses - the live folder and the
  bench agree about what a frame is - and the period estimate is
  re-checked against the recent signal as the watch runs; a small
  drift is adopted as a correction, a large one as a re-lock, and
  both are printed and logged when they happen. The current picture
  is written to a PNG in the session directory (`watch.png`,
  rewritten in place as it improves), and a line of running
  statistics is printed as it goes: frames folded, sync peak, and
  the running image SNR. The watch runs until Ctrl+C and ends with
  a summary - frames folded, duration, final sync and SNR numbers,
  and the paths written. Target selection is exactly capture's:
  `--target N`, an interactive prompt, or explicit `--freq` /
  `--mode` / parameter flags. While watching, the raw IQ is also
  recorded to disk in the session directory by default, so a
  watched session can be re-folded offline afterward and nothing
  is lost; `--no-record` turns the recording off. A file can be
  replayed as a stream instead of the tuner with `--source FILE`
  (`.cu8` or `.cf32`), which is how the live mode is developed and
  proven without hardware. The tuner law is unchanged: before the
  receiver is opened, a holder means refusal with exit code 3 and
  the holder named, never a kill or a preemption; a missing
  `rtl_sdr` or numpy is exit code 2.

Every hunt, capture, reconstruction, and watch is recorded in an
append-only session log (`session.jsonl` in the session directory)
with its parameters and its measured quality numbers.

## Running it

Catoptrics is one file, `catoptrics.py`. The census is standard
library only; reconstruction, simulation, and watch use numpy, and
report unavailable without it. Hunting and capturing need
`rtl_power` and `rtl_sdr` from the rtl-sdr package; watching live
needs `rtl_sdr`.

```
python3 catoptrics.py hunt --low 40M --high 400M
python3 catoptrics.py capture --target 2 --seconds 5
python3 catoptrics.py reconstruct --iq capture.cu8 --target 2
python3 catoptrics.py watch --target 2
python3 catoptrics.py simulate --out sim.cf32
python3 catoptrics.py reconstruct --iq sim.cf32 \
    --mode SIM320x200@60 --rate 11520000 --reference sim.pattern.png
python3 catoptrics.py watch --source sim.cf32 \
    --mode SIM320x200@60 --rate 11520000 --reference sim.pattern.png
```

- `--session DIR` chooses the session directory (default
  `./catoptrics-session`). The target list, session log, and
  default output paths live there.
- `hunt` also accepts `--spectrum FILE` (an rtl_power CSV or a
  plain frequency,dB text file) to census a spectrum recorded
  elsewhere, and `--threshold DB` for the floor-relative peak bar.
- `capture` accepts `--freq` instead of `--target` for a manual
  frequency, plus `--rate`, `--seconds`, and `--gain`. Exit code 3
  means the tuner is held by another program, which is named; exit
  code 2 means a required tool is unavailable.
- `reconstruct` accepts `--mode` (640x480@60, 1024x768@60,
  1280x720@60, 1920x1080@60, or the simulator's SIM320x200@60) or
  explicit `--width/--height/--htotal/--vtotal/--refresh/
  --pixel-clock`. `--reference PNG` scores the result against a
  known pattern with a normalized cross-correlation and writes a
  side-by-side image. Images are written as PNG, encoded with the
  standard library.
- `simulate` writes float IQ (`.cf32`), the source pattern as PNG,
  and a JSON sidecar holding the ground truth: true pixel clock
  (offset from the stated one by `--clock-ppm`, default +37 ppm, so
  sync estimation has something real to find), SNR, and frame count.
- `watch` accepts `--source FILE` to replay a recorded stream,
  `--seconds` to stop after a span of signal, `--out` for the live
  image path, `--interval` and `--every-frames` for the snapshot
  cadence (a snapshot is written when either bound is met), and
  `--no-record` / `--record-out` for the raw stream recording. The
  first snapshot is also kept beside the live image as
  `watch-first.png`, so the earliest formed picture survives the
  rewriting.

## Honest limits

- Modern digital displays fight back. HDMI signalling, spread-
  spectrum clocking, and decent shielding can reduce the recoverable
  image to faint structure or nothing at all. Older and simpler
  signalling leaks more; nothing here promises a readable screen.
- Distance and antenna dominate everything. The difference between a
  ghost and a blank is usually metres and wire, not software.
- The first ghosts are rough. Expect shapes, bands, and the ghosts
  of large text before anything legible, and expect the tool to say
  how rough, in numbers.
- A target is an emanation cluster, not an identified device. Two
  displays in the same mode, phase-locked to the same refresh, can
  present as one family; one display can present spurious peaks the
  census honestly lists as unidentified.
- Live work has its own limits. Sync acquisition needs a few
  frames of signal before the first picture forms - the ghost
  appears after the opening seconds, not at the first sample. The
  live image is a running average of every frame folded so far, so
  a screen that changes faster than the average smears: what watch
  shows is the screen's persistent content, with motion blurred
  into it. Frames folded before a sync correction or re-lock keep
  the grid they were folded on; tracking steers the frames that
  follow, it does not rewrite the average that came before.
- The period estimate carries a small residual bias - measured at
  about +0.1 to +0.25 ppm on the simulator's signal after the
  multi-lag refinement - and a bias integrates: over a long fold
  the accumulated frame starts creep sample by sample away from
  the true ones. On a one-second 0 dB watch this is the difference
  between the correlation's peak (0.993 at 30 frames) and its value
  a half-second later (0.983 at 59 frames); the image SNR keeps
  climbing the whole time (15.7 to 21.0 dB over the same span). At
  lower SNR or over minutes the effect grows. The offline bench,
  which estimates once over the whole capture, lands at the other
  end of the same bias, which is why live and offline agree to
  within about 0.03 of correlation on the same stream rather than
  exactly.

## Verification

The test suite (`python3 -m unittest discover -s tests`, 26 tests)
covers the pattern and codecs, the simulator's ground truth, sync
estimation landing on the true frame period, reconstruction
recovering the test pattern, the census grouping two synthetic
emitter families into two targets in strength order with the shared
harmonic marked ambiguous, target selection driving capture and
reconstruct parameters, reconstruction of a selected simulated
target, and the tuner-busy refusal. The 0.2.0 tests add the live
folder: the final live image against the offline reconstruction of
the same stream (normalized cross-correlation at or above 0.95),
snapshot improvement over time within the tolerance stated in the
test, stable tracking across the planted +37 ppm clock offset with
no re-lock, a re-lock adopted after a mid-stream clock jump to
-800 ppm with the tracked period landing on the new clock, the
recorded stream re-folding offline to the same picture, and the
tuner-busy and unavailable refusals on the watch path.

The captured 0.2.0 watch proof replays the simulator's emanation as
a stream (no tuner). At the standard point - SIM320x200@60, 12 dB
SNR, true clock +37 ppm off the stated one - watch locks at a frame
period of 191992.9 samples with an autocorrelation peak of 0.905,
folds 17 frames, and finishes at a normalized cross-correlation of
1.000 against the source pattern with a measured image SNR of
29.8 dB; the final live image against the offline reconstruction of
the same stream scores 1.000. On a one-second 0 dB stream the
correlation forms as 0.989 at 15 frames, 0.993 at 30, 0.991 at 45,
and 0.983 at 59, while the image SNR climbs 15.7 to 21.0 dB; the
offline reconstruction of that stream scores 0.992, and live
against offline is 0.998. In both runs the recorded stream is
byte-identical to the stream that was replayed. The offline bench
numbers stand as measured at 0.1.0 and re-measured here: the 12 dB
pattern reconstructs at a correlation of 0.999 to 1.000 with a sync
peak of 0.916. Hardware verification against a live display is
pending a free tuner and will be recorded here when it happens.

## State

0.2.0 - the live bench. The `watch` command: incremental folding
from a live or replayed stream, a live PNG with running statistics,
sync corrections and re-locks reported as they happen, and the raw
stream recorded by default for offline re-folding. The frame-period
estimator gains a multi-lag refinement, shared by the live folder
and the offline bench.

0.1.0 - the first release. The program, the test suite, and this
README.
