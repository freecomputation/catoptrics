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

## What it does

- `hunt` sweeps a band and ranks candidate emanation frequencies:
  measured peaks scored against their local noise floor, annotated
  when a peak sits near a harmonic of a known display pixel clock.
- `capture` records raw IQ from the receiver, with its parameters
  written into the session log before the first sample is taken.
- `reconstruct` is the bench: envelope detection, frame-sync
  refinement by autocorrelation, folding into a raster, averaging
  across frames, and a written report of sync quality and image SNR.
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
