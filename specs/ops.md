# Ops

## Purpose
Every operation on Values. Each op creates a new Value `out` with its
parents in `_prev`, its label in `_op`, and its own backward rule in
`_backward` (see specs/value.md). Add is specified in specs/add.md.

## The one pattern
Every backward rule is:

    input.grad += (local gradient) * out.grad

- Local gradient = ∂out/∂input, this op's own derivative, computed from
  numbers saved in the forward pass (`.data`), never from Value objects.
- `* out.grad` is the chain rule step: without it, the input's gradient
  would ignore everything after this node.
- `+=`, never `=`: an input used in several places gets gradient from each.
  Example: `out = a * a` with a = 3. Line 1 adds 3, line 2 adds 3, so
  a.grad = 6 = 2a. With `=`, line 2 overwrites line 1 and gives 3, which is
  silently wrong (no crash).

## Ops

| op | forward | backward | `_op` |
|---|---|---|---|
| `a + b` | a.data + b.data | `a.grad += 1 * out.grad`, `b.grad += 1 * out.grad` | `"+"` |
| `a * b` | a.data * b.data | `a.grad += b.data * out.grad`, `b.grad += a.data * out.grad` | `"*"` |
| `a ** n` | a.data ** n | `a.grad += n * a.data**(n - 1) * out.grad` | `"**n"` |
| `a.tanh()` | tanh(a.data) | `a.grad += (1 - out.data**2) * out.grad` | `"tanh"` |
| `a.exp()` | e ** a.data | `a.grad += out.data * out.grad` | `"exp"` |
| `a.relu()` | a.data if a.data > 0 else 0 | `a.grad += (out.data > 0) * out.grad` | `"relu"` |

Notes from the derivations:
- **power:** `n` is a plain int/float, never a Value. Only Values get
  gradients; plain numbers are constants. If n were a Value we'd need
  ∂out/∂n = aⁿ·ln(a), which breaks for a ≤ 0. In practice exponents are
  always fixed (x**2, x**-1). `ln` only appears when the variable is in the
  exponent, so it does not appear here.
- **tanh, exp:** reuse `out.data` (saved in forward) instead of recomputing
  tanh/exp in backward.
- **Python:** use `**` for powers. `^` is XOR in Python.

## Edge cases
- **Plain numbers on either side** (`a * 2`, `2 * a`, same for `+`): wrap
  the plain number in a Value first, so the backward rule can read
  `.data` unchanged. The wrapped number is a leaf; it gets a gradient that
  nobody reads. `2 * a` needs `__rmul__` (Python tries int's `__mul__`
  first, which fails, then falls back to `a.__rmul__(2)`), like `__radd__`.
- **ReLU at exactly 0:** the slope is undefined, so we pick **0** (the
  flat side). This matches PyTorch, so comparison tests don't fail on an
  input of exactly 0. The rule `(out.data > 0)` already gives 0 there.
  Side effect to remember for Day 2: a ReLU whose input is negative for
  every example always gets gradient 0 and stops learning ("dead ReLU").

## Tests I want
- `a * a` with a = 3 → a.grad = 6.
- `2 * a` and `a * 2` with a = 3 → both give data 6 and a.grad = 2.
- `a ** 2` at a = 3 → a.grad = 6; `a ** 3` at a = 3 → a.grad = 27.
- relu at a = 2 → a.grad = 1; at a = -2 → 0; at a = 0 → 0.
- More tests chosen in Block 5, checked against PyTorch.
