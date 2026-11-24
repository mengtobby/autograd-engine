# Gradcheck

## Purpose
Check the engine's gradients using nothing but the definition of a gradient:
nudge an input, rerun the forward pass, and see how much L moves. Unlike
PyTorch, this reference shares no code or assumptions with my specs, so it
catches mistakes in my understanding, not just in the code.

## Interface
`gradcheck(f, inputs, h=1e-6, tol=1e-6) -> None`
- `f`: a function that takes a list of Values and returns L. Gradcheck
  makes fresh Values from `inputs` for every call, so it can read the
  input Values' `.grad` after `backward()`.
- `inputs`: a list of floats to check at.
- Raises AssertionError naming the input index, engine grad, estimate and
  error when any input fails.

## Method
1. Build L from `inputs` with fresh Values, call `L.backward()`, and record
   each input's `.grad` (the engine's answer).
2. For each input i, compute the **central difference** with forward passes
   only, nudging input i and leaving the others alone:

       estimate_i = ( L(xᵢ + h) − L(xᵢ − h) ) / (2h)

3. Pass if  `| engine_i − estimate_i |  ≤  tol × (1 + | estimate_i |)`.

## Why central, not one-sided
Central uses the same nudge, but with x in the middle, so the error from the
curve's bend cancels out on both sides. Error shrinks like h² instead of h.
(Block 3 graph, h = 0.01: one-sided −23.91, central −24.00, true −24.)

## Why h = 1e-6
Measured on `exp(tanh(2a))·a` at a = 0.7, error vs h: 8.5e-3 at 1e-1, a
minimum of 1.0e-10 at 1e-6, back up to 4.3e-3 at 1e-14. A tiny h magnifies the
rounding error of the computer: L(x+h) and L(x−h) share ~14 of their 16 digits,
so their difference is mostly rounding noise, and dividing by a tiny h blows it
up. 1e-6 is the sweet spot (also PyTorch's default). The exact sweet spot
moves a little per function, so it's a default, not a law.

## Why this tolerance
The tolerance sits in the gap between noise and bugs: a correct engine's
error is ~1e-10 to 1e-8; Block 6 bugs were off by whole units.
- 1e-12 would flag correct code (noise is bigger).
- 0.1 would let a subtle bug (off by 0.05) through.

Rounding noise is relative to the size of the numbers, so a fixed 1e-6 would
falsely flag a gradient of 50,000. The allowed error therefore scales with
the gradient's size (relative error), plus a fixed floor about the size of
the rounding noise so a true gradient of 0 doesn't get an allowed error of 0:

| gradient size | allowed error |
|---|---|
| 0 | 1e-6 |
| 3 | 4e-6 |
| 50,000 | 0.05 |

## Edge cases
- **Kinks:** relu at exactly 0 gives estimate (h − 0)/2h = 0.5 but the engine
  gives 0. The central difference straddles the kink and averages the two
  slopes. Never gradcheck at a kink: keep relu inputs at least 1e-3 away from 0.
- **Reused inputs:** an input used several times in f still gets one estimate
  (nudging it moves every use), so gradcheck checks `+=` for free.

## Tests I want
- Every op passes gradcheck at a few inputs away from kinks: +, *, **n (n = 2,
  3, −1, 0.5 with a > 0), tanh, exp, relu (at ±0.5, ±2), and derived −, /.
- The Block 3 graph and the tanh/exp chain pass.
- Self-test: gradcheck must **fail** on a deliberately wrong gradient (e.g. a
  function whose `_backward` is broken), so we know the checker can catch bugs.
- Self-test: relu at exactly 0 fails, confirming the kink warning.
