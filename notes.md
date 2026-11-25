# Notes

Predictions, results, surprises. One entry per piece.

Entry format:

```
## <date> — <piece>
Prediction:
Result:
Surprise / what I got wrong:
One-line takeaway:
```

---

## 2026-11-20 — hand trace (Day 1, Block 3)
a = 2, b = -3, c = 10; e = a*b, d = e + c, L = d*d
Prediction: forward e = -6, d = 4, L = 16.
Backward L.grad = 1, d.grad = 8, e.grad = 8, c.grad = 8, a.grad = -24, b.grad = 16.
Result: all correct. Engine and PyTorch both give a.grad = -24, b.grad = 16,
c.grad = 8 (tests/test_engine.py, 24/24 passed).
Surprise / what I got wrong: twice did `4 += 4` as 16. `+=` adds the new
piece to what's there (4 + 4 = 8). Wasn't sure how add passes gradient: it
passes it through unchanged to every input.
One-line takeaway: `d * d` uses d twice, so d collects 4 + 4 = 8 = 2d.

## 2026-11-24 — gradcheck (Day 2, Block 1)
Estimate a gradient from forward passes only: nudge a by h, a.grad ≈ (L₁ − L₀)/h.
Block 3 graph, h = 0.01: one-sided gives -23.91 (true -24); central gives -24.00.

**Q: Why central differences rather than one-sided?**
A: Central uses the same nudge, but with a in the middle, so the error from
the curve's bend cancels out on both sides. (Error shrinks like h² instead of h.)

**Too-small h** (L = exp(tanh(2a))·a at a = 0.7, central difference vs engine)
Prediction: (B) error goes down at first, then back up at very small h.
Result: confirmed. Error 8.5e-3 (h=1e-1) → 1.0e-10 at h=1e-6 (best) →
back up to 4.3e-3 at h=1e-14.
Surprise / what I got wrong: didn't guess where the minimum was; it's ~1e-6.

**Q: Why does a too-small h make gradcheck worse?**
A: A tiny h magnifies the rounding error of the computer. (L(a+h) and L(a−h)
share ~14 of their 16 digits, so the difference is mostly rounding noise, and
dividing by tiny h blows it up.) Two errors fight: the curve's bend (shrinks
with h) vs rounding noise / h (grows as h shrinks).
One-line takeaway: smaller h is not always better; ~1e-6 is the sweet spot.

## 2026-11-25 — gradient shapes (Day 3, Block 3)
Prediction (as given):
- out = X (32, 10) + b (10,): "b.grad has the same shape as b ... possibly
  (32, 10)". ✗ Mixed up b with out. b.grad is (10,) = G.sum(axis=0): the 32
  copies get summed back into one row.
- H = X (32, 784) @ W (784, 128): H is (32, 128) ✓; W.grad same shape as W,
  (784, 128) ✓ (formula not given); X.grad same as X, (32, 784) ✓.
  Follow-up: W.grad = Xᵀ @ G = (784, 32) @ (32, 128) = (784, 128) ✓. The inner
  32 is the batch: matmul sums over the examples, like the bias does.
Result: (Block 5, check in the engine)
Surprise / what I got wrong: confused the broadcast input's shape with the
output's shape.
One-line takeaway: the gradient matches the input's own shape, never the
broadcast output's.

## 2026-11-24 — training loop (Day 2, Block 3)
Step order: forward + loss → backward → update → zero grads (C A B D).

**Moons training** (prediction, Day 2 Block 4)
Prediction: loss curve drops fast early, then flattens (B). Final accuracy
90%+ on the 100 points.
Result: confirmed. Loss 0.455 → 0.065, sharp drop in the first ~5 steps
then steady flattening. Accuracy 84% → 99% (1 point on the boundary).
Boundary plot: clean curve weaving between the crescents.
Surprise / what I got wrong: the random untrained net already scored 84%;
most training went into the hard points near the crescent tips.
One-line takeaway: the engine trains a real network; loss falls fast, then
the last few % take most of the steps.

**Experiments** (Day 2 Block 6; baseline lr 0.1, [16, 16, 1], tanh, 99%)
Prediction:
- E1 lr: 0.001 learns too slowly; 1.0 learns well; 10 jumps around / blows up.
- E2 depth: one hidden layer [16, 1] beats no hidden layer [1].
- E3 ReLU: worse than tanh.
Result (experiments/moons_day2.py):
- E1: lr 0.001 → 85% (loss 0.314, too slow ✓); lr 1.0 → 100%, loss 0 (✓,
  better than baseline); lr 10 → loss jumped to 22.9, ended 4.47, 68% ✓.
- E2: no hidden [1] → 86%; one hidden [16, 1] → 89% ✓ (both well below
  [16, 16, 1]'s 99% after 100 steps).
- E3: ReLU → 99%, about the same as tanh ✗.
Surprise / what I got wrong: ReLU wasn't worse; on a problem this small both
can bend the boundary enough.
One-line takeaway: lr has a sweet spot (too small crawls, too big overshoots).

**Q: Why can't a network with no hidden layer separate moons?**
A: Its score w₁x + w₂y + b = 0 is a straight line; the crescents interlock,
so any line cuts through the tips (86% is about the best a line can do).
The hidden layer's nonlinearity lets the boundary bend. Without tanh/ReLU,
stacked layers are a weighted sum of weighted sums: still one straight line.

**No zero grads** (prediction)
Prediction: grads from previous steps leak into every later step (`+=`
keeps adding: 5 then 5 + 3 = 8 instead of 3), so updates use the wrong,
growing gradient. Training: (C) loss jumps around or blows up, accuracy poor.
Result: ✗ it reached 100% with loss 0. But the largest weight grew
1.18 → 3.11 → 5.79 (steps 10/50/99) vs ~1.1 with zeroing, and a leftover
gradient of ~0.73 kept pushing even at loss 0.
Surprise / what I got wrong: the bug made results look better. The summed
gradients acted like a growing lr early (lr 1.0 also did well), and with
hinge loss, drifting weights just widen the margins. Runaway, but hidden.
One-line takeaway: a bug can improve the metrics; only a direct test
(grads are 0 after a step) catches it. Check the mechanism, not the outcome.

## 2026-11-24 — Day 1 drill redo, cold (Day 2, Block 9)
Recalled: data (float), grad starts at 0, parents, op; wrap plain numbers.
Not recalled: _backward field, __rmul__, add/multiply backward, the 3 steps
of backward(), the 3 rules. Not built (would fail like round 1).
Still mixing up: "starts at 0" (every node at creation) vs "seed = 1"
(backward sets L.grad).
Memory hooks: add copies; multiply swaps (other input's data × out.grad);
Build, Seed, Walk back.
TODO next session, cold, before Day 3: multiply backward, Build-Seed-Walk
back, the three rules.

Retry right after reviewing (not cold): recalled fields, wrapping, add
backward, multiply backward with the swap (other input's .data), Build →
Seed → Walk back, all three rules. Follow-ups: build = "visiting the
parents" (missed: then append, and the visited check); walk back calls each
node's _backward ✓; named _as_value instead of __rmul__.
Literal build (drills/day2_engine.py): 5/9 pass (Day 1 round 1: 0/9).
Failures = the two gaps: 2 * a (no __rmul__); shared node + Block 3 (no
visited check → node listed twice → _backward runs twice, the same bug
I found in the bug hunt).
Remaining weak spots: the visited check in build, __rmul__. (Day 3 cold check
dropped at my request; no from-memory drill on Day 3.)

Bug hunt (Block 7): accepted as is (found 1 of 2 alone), no rerun.

## 2026-11-24 — 2-layer trace (Day 2, Block 8)
x = 2, w₁ = 0.5, b₁ = 0, h = relu(z), w₂ = −1, b₂ = 0.5, y = +1, hinge loss.
Forward: z = 1, h = 1, s = −0.5 (wrong side), L = 1.5. (First called s 1.5;
that was L.)
Backward: s.grad = −1 (mine), b₂ −1, w₂ −1, h +1 (Claude did this layer),
z 1, w₁ 2, b₁ 1 (mine). All match the engine.
One step at lr 0.1: s → +0.15 (now correct), L → 0.85.
Surprise / what I got wrong: h.grad = +1: the negative weight w₂ flips the
sign for everything below it ((−1) × (−1)).
One-line takeaway: each layer is the same two rules (multiply: other input's
data × grad; add: pass through), and signs carry through the weights.

## 2026-11-24 — bug hunt in engine.py (Day 2, Block 7)
Tests: 9 failed, 45 passed.
Bug 1 (found with an explanation): tanh backward dropped `× out.grad`.
Clue: gradcheck on tanh alone passed (out.grad = seed = 1, so the missing
multiply is invisible) but tanh inside a chain failed. Fixing it left 5.
Bug 2 (found myself): `if node in visited: return` deleted from build().
Clue: the reused node's own grad was right (d = 8) but the node below got
double (e = 16 = 8 + 8): d was in the list twice, so its _backward ran twice.
a*a still passed because a leaf's _backward does nothing, even twice.
Surprise / what I got wrong: couldn't see what a single-op test can't catch.
One-line takeaway: compare what passes with what fails; a single-op test
can't catch a missing `× out.grad`, and fixing one bug at a time shows what's
left.

## 2026-11-20 — spec from memory (Day 1, Block 7)
Read the model spec first, then recalled from an outline (easier than intended).
Round 1 result: literal build failed 9/9. Got right: the four fields, wrap
plain numbers, add forward + backward with +=, multiply forward, "walk the
list in reverse".
Missed: grad starts at 0; leaf _backward does nothing; __radd__/__rmul__;
multiply backward (wrote `b.grad += a.grad * out.grad`, no line for a);
how the list is built; the seed; calling _backward at each node.
Round 2: fixed grad -> data in multiply; recalled the seed L.grad = 1 (but
mixed it up with grad's starting value 0). Claude filled the rest; 9/9 pass.
Surprise / what I got wrong: grad vs data in the multiply rule again, and
"starts at 0" (every node, at creation) vs "seed = 1" (only L, when
backward begins).
One-line takeaway: the backward walk is: build list (parents first) ->
seed L.grad = 1 -> reverse the list and call each _backward.
TODO: redo this drill cold at the start of next session, before Day 2.

## 2026-11-20 — backward: break it predictions (Day 1, Block 6)

**Skip the topo sort (run `_backward` in forward order instead of reversed)**
Prediction (worked through with help): a single op (`L = a * b`) is still
right by luck, because the leaves run first but do nothing, and L runs with
its grad already seeded. Any graph with a middle node breaks: in the Block 3
trace, e and d run before their grads arrive and pass on 0. Only d.grad = 8
is right; e, c, a, b all 0. Tests that should still pass: single-op ones
(a*a, a+a, pow, relu, 2*a, leaf/seed tests).
Result: confirmed. 10 failed, 14 passed. Block 3 trace gave d = 8 and
e, c, a, b = 0, exactly as predicted. All single-op tests passed.
Surprise / what I got wrong: `a - b`, `3 - a`, `a / b`, `3 / a` and
`a + b + c` failed but `-a` passed. Derived ops look like one op, but
`a - b = a + (b * -1)` builds two nodes, with a middle node that runs too early.
One-line takeaway: skipping the sort only works when every input to the
output is a leaf; one line of code can be several nodes.

**Don't set the output grad to 1**
Prediction: every gradient comes out 0. Each backward step multiplies by the
grad of the node after it, so a 0 seed passes 0 down the whole chain.
Weights never change, so nothing learns, and nothing crashes (silent).
Also: relu at -2 and at 0 (expected grad 0) will still pass by chance, since
the broken engine gives 0 for everything. relu at 2 should catch it.
Result: confirmed. 19 failed, 5 passed. relu at -2 and 0 passed by chance;
relu at 2 failed. The other 3 passes never call backward().
Surprise / what I got wrong: nothing; both predictions held.
One-line takeaway: a test that expects 0 can't tell "correctly 0" from
"nothing flowed", so always pair it with a nonzero case.

**`=` instead of `+=`**
Prediction: any value used more than once gets only its last use's gradient.
`a * a` at a = 3 gives a.grad = 3 instead of 6. Tests that catch it: `a * a`,
`a + a`. No crash, just a wrong number.
Result: 6 failed, 18 passed, no crash. a*a gave 3, a+a gave 1 (as predicted);
shared node gave (3, 6) instead of (7, 14); Block 3 trace gave d.grad 4,
a.grad -12, b.grad 8; tanh/exp chain failed (a used twice in it).
Surprise / what I got wrong: only named the obvious tests. Reused values were
hiding in the Block 3 trace (d*d) and the tanh/exp chain (a used twice). And
the error didn't stay at d: everything below d halved, because each node
multiplies from the grad above it.
One-line takeaway: one `=` at a shared node silently corrupts every gradient
between it and the inputs.
