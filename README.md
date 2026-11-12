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

Hunt, read the list, select a target, capture, reconstruct.

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

Every hunt, capture, and reconstruction is recorded in an append-only
session log (`session.jsonl` in the session directory) with its
parameters and its measured quality numbers.

## Running it

Catoptrics is one file, `catoptrics.py`. The census is standard
library only; reconstruction and simulation use numpy, and report
unavailable without it. Hunting and capturing need `rtl_power` and
`rtl_sdr` from the rtl-sdr package.

```
python3 catoptrics.py hunt --low 40M --high 400M
python3 catoptrics.py capture --target 2 --seconds 5
python3 catoptrics.py reconstruct --iq capture.cu8 --target 2
python3 catoptrics.py simulate --out sim.cf32
python3 catoptrics.py reconstruct --iq sim.cf32 \
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

## Verification

The test suite (`python3 -m unittest discover -s tests`, 18 tests)
covers the pattern and codecs, the simulator's ground truth, sync
estimation landing on the true frame period, reconstruction
recovering the test pattern, the census grouping two synthetic
emitter families into two targets in strength order with the shared
harmonic marked ambiguous, target selection driving capture and
reconstruct parameters, reconstruction of a selected simulated
target, and the tuner-busy refusal.

The simulator proof, run at release: the SIM320x200@60 pattern at
12 dB SNR with the true clock +37 ppm off the stated one reconstructs
with a normalized cross-correlation of 0.999 against the source
pattern, an autocorrelation sync peak of 0.916, and a measured image
SNR of 28.8 dB over 17 averaged frames. At 6 dB SNR the correlation
is 0.997; at 0 dB with a one-second capture it is 0.960, with the
sync peak falling to 0.592 - the bench degrades in the open, in
numbers. Hardware verification against a live display is pending a
free tuner and will be recorded here when it happens.

## State

0.1.0 - the first release. The program, the test suite, and this
README.
