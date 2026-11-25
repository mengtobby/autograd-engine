# Tensor

## Purpose
The scalar engine makes one Value per number: too slow for MNIST (~100,000
weights, 784 inputs per image). A Tensor node holds a whole NumPy array, so
"multiply a (32, 784) batch by a (784, 128) weight matrix" is one node, and
NumPy does the arithmetic. Everything from Days 1–2 carries over unchanged:
`data`, `grad`, parents, `_backward`, `+=`, Build → Seed → Walk back,
gradcheck. What's new is shapes.

## Interface
`Tensor(data)`: `data` is anything `np.array` accepts, stored as
**float64**. Fields as in specs/value.md: `data` (ndarray), `grad`
(ndarray of zeros, same shape), `_prev`, `_backward`, `_op`.
Plain numbers and arrays on either side of an op are wrapped in a Tensor.

**Why float64:** gradcheck uses h = 1e-6. float32 keeps ~7 significant
digits, so L(x+h) − L(x−h) would be almost all rounding noise and gradcheck
would fail on correct code.

## The shape rule
**A gradient always has exactly the same shape as its data.** `.grad`
answers "nudge this one number, how much does L move?", one answer per
number. L is always a single number (the loss).

## Ops
| op | forward | backward (G = out.grad) |
|---|---|---|
| `A + B` | elementwise (broadcasts) | `A.grad += unbroadcast(G)`, same for B |
| `A * B` | elementwise (broadcasts) | `A.grad += unbroadcast(B.data * G)`, `B.grad += unbroadcast(A.data * G)` |
| `A ** n` (n plain number) | elementwise | `A.grad += n * A.data**(n-1) * G` |
| `tanh`, `exp`, `relu` | elementwise | Day 1 rules, elementwise: `(1 - out.data**2) * G`, `out.data * G`, `(out.data > 0) * G` |
| `-A`, `A - B`, `A / B` | built from `*`, `+`, `**` as in specs/backward.md | free |
| `A @ B` | matrix multiply, 2-D only | `A.grad += G @ B.data.T`, `B.grad += A.data.T @ G` |
| `A.sum(axis=None, keepdims=False)` | NumPy sum | copy G back to A's shape along the summed axes |
| `A.mean(axis=None, keepdims=False)` | sum ÷ n | same as sum, divided by n (the number of entries averaged) |
| `A.reshape(*shape)` | same numbers, new shape | `A.grad += G.reshape(A.shape)` |
| `A.T` | transpose (2-D) | `A.grad += G.T` |

## Matmul, derived from shapes
C = A @ B with A (2, 3), B (3, 4): C and G are (2, 4).
- A.grad must be (2, 3). From G (2, 4) and Bᵀ (4, 3): `G @ Bᵀ` = (2, 3) ✓.
  `G @ B` is (2, 4) @ (3, 4): impossible.
- B.grad must be (3, 4). From Aᵀ (3, 2) and G: `Aᵀ @ G` = (3, 4) ✓.
  `G @ Aᵀ` is impossible. Order matters: matmul isn't commutative.
It's Day 1's "multiply swaps" upgraded: each input gets the other input
times G; shapes decide the transpose and the side. With 1×1 matrices it is
exactly `a.grad += b.data * out.grad`.

## Broadcasting: unbroadcast
**Broadcasting copies forward, so backward sums over the copies, back to the
original shape.** It's `+=` from Day 1: each copy is a separate use.
- b (2,) added to X (3, 2): b is copied down the rows; G = [[1,2],[3,4],[5,6]]
  → b.grad = column sums [9, 12] (`G.sum(axis=0)`).
- c (3, 1) added to X (3, 2): c is copied across the columns → c.grad = row
  sums [[3], [7], [11]] (`G.sum(axis=1, keepdims=True)`).

`unbroadcast(G, shape)`:
1. **Extra dimensions:** if the input had fewer dimensions than G, sum away
   the extra leading ones.
2. **Stretched dimensions:** wherever the input had size 1 but G is bigger,
   sum along that axis and keep it as size 1.
Result has exactly the input's shape.

In a network, a bias (10,) added to a batch (32, 10) gets the sum of what
all 32 examples want.

## Sum is the mirror of broadcasting
| forward | backward |
|---|---|
| broadcast (copy) | sum over the copies |
| sum | copy G back to every entry |
A (2, 3), out = A.sum() = 21, G = 1: nudging any entry by 0.01 moves out by
0.01, so A.grad is a (2, 3) array of ones.

## Reshape and transpose
Numbers only move, nothing is computed, so the gradient moves back: reshape
G to A's shape; transpose G back.

## backward(): iterative topological sort
Same Build → Seed (L.grad = ones, shape of L, which must be a single number)
→ Walk back, but Build uses a **loop with an explicit stack** (a plain Python
list), not recursion.

**Why:** recursive build calls itself once per level of depth. Python stops
at ~1,000 nested calls and crashes with RecursionError, and a chain of 5,000
nodes (realistic in training) goes 5,000 deep. A loop keeps the same to-do
list in a plain list we manage, which can hold millions of items. Output is
the same: parents before the node, each node once (visited set).

## Tests I want
- Gradcheck **every op** (tensor version: nudge each entry one at a time,
  same h = 1e-6 and tolerance rule as specs/gradcheck.md), on random inputs
  away from relu's kink.
- At least 3 broadcasting cases: (3, 2) + (2,), (3, 2) + (3, 1), and
  (3, 2) + a plain number. Exact values for the two worked examples above
  ([9, 12] and [[3], [7], [11]]).
- Matmul matches PyTorch for A (2, 3) @ B (3, 4).
- Deep graph: a chain of 5,000 ops runs backward without RecursionError and
  gets the right gradient.
- Sum, mean (with and without axis), reshape, transpose: shapes and values.
