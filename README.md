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
- `capture` records raw IQ from the receiver. It takes a target by
  number (`--target N`, from the last hunt) or asks for one: run in
  a terminal with no target named, it prints the list and reads the
  operator's choice. Selecting a target locks the capture frequency
  and the expected mode - pixel clock, resolution, refresh - to that
  target's measured parameters, and the session log records which
  target was selected.
- `reconstruct` is the bench: envelope detection, frame-sync
  refinement by autocorrelation, folding into a raster, averaging
  across frames, and a written report of sync quality and image
  SNR. It accepts the same `--target N`, which supplies the mode
  parameters the target was measured with.
- `simulate` synthesizes a raster emanation from a built-in test
  pattern, so the bench can be proven end to end without hardware.

Every hunt, capture, and reconstruction is recorded in an append-only
session log (JSONL) with its parameters and its measured quality
numbers.

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

## State

Founding. This README, the design above, and the license. The
program follows.
