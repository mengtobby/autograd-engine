# Backward

## Purpose
`L.backward()` fills in `.grad` for every node in the graph that produced L:
how much L changes per unit nudge of that node. It only computes gradients;
it never changes any `.data`. Updating weights is the training loop's job
(Day 2: `w.data -= lr * w.grad`).

Comparing a prediction with a target happens in the **forward** pass, inside
the loss formula. By the time backward runs, L is just a number, and
backward only asks how L responds to each node.

## Interface
`Value.backward(self) -> None`, called on the final output (the loss).

## Steps
1. **Seed:** set `L.grad = 1`.
2. **Order:** build a topological order of every node reachable from L
   (each node listed after its inputs), then reverse it.
3. Visit the nodes in that reversed order and call `node._backward()`
   (its own rule from specs/ops.md) once on each.

### Building the topological order
Depth-first from L, with a `visited` set: to visit a node, skip it if
already visited, otherwise mark it, visit each of its parents (`_prev`),
then append the node to the list. A node is appended only after all its
parents, so the list is topological. The `visited` set makes a node that is
used twice appear once. (Recursive is fine on Day 1; Day 3 explains why the
tensor engine needs an iterative version.)

## Why reverse topological order
**Rule: a node's `_backward` may only run after every node that *uses* it
has already run.** A node's gradient comes from the nodes that use it. If it
passes its gradient on before they've all delivered, it passes on an
incomplete number, and since each `_backward` runs once, nothing goes back to
fix it. No crash, just a wrong gradient.

- Example 1 (p before d): `d = p + (-4)` uses p. If p runs first, p.grad is
  still 0, so w.grad gets 0 instead of 8.
- Example 2 (*every* user): x is used by e and f, correct x.grad = 3 + 4 = 7.
  Order L → e → x → f: x runs with only e's 3, so w.grad = 2·3 = 6 instead
  of 14.

Topological order lists each node after its inputs (the forward order).
Reversing it lists each node after everything that uses it, which is the
rule above.

## Why the seed is 1
`L.grad` means: nudge L by 0.01, how much does L change? By 0.01, so
∂L/∂L = 1. Every other gradient is built by multiplying local gradients onto
the grad of the node after it, all the way back to `L.grad`, so L is the seed
everything grows from. If it stayed 0, every gradient would be 0, weights
would never change, and nothing would crash.

Every other grad starts at 0 because nothing has been added to it yet. That's
the empty starting value for `+=`, not a sign that the node is uninvolved.

## Why `+=`
A node used in several places gets gradient from each use. `a * a` at a = 3:
both slots of the multiply add 3, so a.grad = 6 = 2a. With `=`, the second
overwrites the first and gives 3.

## Derived ops (no new backward rules)
Built only from existing ops, so their gradients come for free from the
rules in specs/ops.md:

| op | built as | Python methods |
|---|---|---|
| `-a` | `a * (-1)` | `__neg__` |
| `a - b` | `a + (b * (-1))`, i.e. `a + (-b)` | `__sub__`, `__rsub__` (for `3 - a`) |
| `a / b` | `a * (b ** -1)` | `__truediv__`, `__rtruediv__` (for `3 / a`) |

Example: for `a / b`, the power rule gives `b ** -1` local gradient
`(-1) · b⁻²`, and the multiply rule applies on top, giving
∂(a/b)/∂b = −a/b² (the quotient rule) without anyone writing it.
Fewer hand-written rules means fewer places for bugs.

## Worked example
x = 2, w = 3, target 4. p = w * x = 6, d = p + (-4) = 2, L = d ** 2 = 4.
Backward: L.grad = 1 → d.grad = 2·2·1 = 4 → p.grad = 1·4 = 4 →
w.grad = 2·4 = 8, x.grad = 3·4 = 12.
Check: L = (2w − 4)², dL/dw = 4(2w − 4) = 8 at w = 3. ✓

## Tests I want
- Seed: after `L.backward()` on a leaf L, L.grad = 1.
- The worked example: w.grad = 8, x.grad = 12.
- Shared node: w = 1, x = w*2, e = x*3, f = x*4, L = e + f → x.grad = 7,
  w.grad = 14.
- Derived ops with a = 6, b = 2: `-a`, `a - b`, `3 - a`, `a / b`, `3 / a`.
  Data and grads match PyTorch (for `a / b`: a.grad = 1/b = 0.5,
  b.grad = −a/b² = −1.5).
