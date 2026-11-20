# Add

> Worked example: the one spec written for you. Copy its shape for every other spec.

## Purpose
out = a + b for Value objects; also accepts plain numbers.

## Interface
`Value.__add__(self, other) -> Value`, `Value.__radd__(self, other) -> Value`.
`other` may be a Value or an int/float (wrap it in a Value).

## Forward
out.data = a.data + b.data; out's parents are (a, b).

## Backward
∂out/∂a = 1, ∂out/∂b = 1, so a.grad += out.grad and b.grad += out.grad.
Use += because a may feed into several nodes.

## Edge cases
a + a: a.grad should get 2 × out.grad. 3 + a: needs __radd__.

## Tests I want
b = a + a with a=3 → a.grad = 2. Compare d = a + b + c against PyTorch.
