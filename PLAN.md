# Stage 1 plan: autograd from scratch (AI-native)

Four days: scalar engine → gradcheck + MLP → tensor engine → softmax CE + MNIST.

**Division of labor:** I never write files by hand. I do the thinking:
derivations, design decisions, predictions, diagnosis and review. Claude
writes every file (specs, code, tests, notes, commits) from what I say,
and checks my understanding before we move on.

The skill being trained: understand the math well enough to **direct**,
**verify** and **debug** AI-written code, not to type it.

## The loop for every piece

1. **Derive (in chat)**: Claude asks leading questions; I work the math out. No answers before I try.
2. **Dictate the spec**: I make the decisions (interface, backward rules, edge cases, tests). Claude writes `specs/<piece>.md` in my words, asks about anything I left out instead of filling it in, and I approve it.
3. **Predict**: I state predictions; Claude logs them in `notes.md` *before* anything runs.
4. **Generate**: Claude implements the approved spec.
5. **Review**: Claude walks through the code; I explain 2+ lines Claude picks; I approve or push back.
6. **Verify**: Claude runs the tests. On a failure, I diagnose before any fix.
7. **Break**: I choose a deliberate change and predict the result; Claude applies it and runs it.
8. **Commit**: I say what I learned; Claude writes the commit.

## Recall drills (replace hand-writing)

Same understanding, tested without typing code:

- **Spec from memory**: one message, no notes, describing a component precisely enough to build. Claude implements it *literally* in `drills/`, filling no gaps, and runs the real test suite against it. Missing or wrong details show up as failing tests.
- **Explain-back**: I explain a concept as if teaching it; Claude probes until it's airtight.
- **Trace**: Claude gives a tiny graph with real numbers; I compute forward and backward in chat; the engine checks me.
- **Blind review**: Claude shows code with 0–2 bugs (count unknown); I find them by reading alone, then confirm with tests.

---

## Day 1: Scalar engine

Goal: a `Value` class whose gradients match PyTorch.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 45 min | Derive in chat: local gradients for +, ×, **n, tanh, exp, ReLU | [x] |
| 2 | 30 min | Dictate specs: value.md (fields), ops.md (all ops), backward.md | [x] |
| 3 | 15 min | Trace: hand-compute every gradient for a=2, b=-3, c=10, d=a*b+c, L=d*d | [x] |
| 4 | 45 min | Claude generates engine.py; review, explain 2 lines | [x] |
| 5 | 30 min | Choose tests: that expression, a + a, a * a, a tanh/exp chain, all vs. PyTorch; Claude writes them; I say what one test proves | [x] |
| 6 | 20 min | Break it (below) | [x] |
| 7 | 30 min | Spec from memory: Value with + and × plus backward(), no notes; literal build in drills/ must pass the Day 1 tests | [~] done with help; redo cold before Day 2 |

backward.md must answer (my words):
- Why the output's grad starts at 1.
- Why topological order, and why reversed.
- Why += rather than =.
- How subtraction, division and negation come from existing ops.

Break it (predict first):
- `=` instead of `+=`: which test fails?
- Skip the topological sort: when does it still give the right answer by luck?
- Don't set the output grad to 1: what do all the gradients become?

**Done when:** all tests pass, my trace matches the engine, and the drill build passes.

---

## Day 2: Gradcheck and MLP

Goal: prove the engine is correct in general, then train a network with it.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 30 min | Dictate gradcheck.md: central differences, choice of h and tolerance, and why | [x] |
| 2 | 30 min | Dictate nn.md: Neuron, Layer, MLP, parameters(), weight init | [x] |
| 3 | 20 min | Dictate train_moons.md: forward, loss (hinge or MSE), zero grads, backward, SGD step | [x] |
| 4 | 15 min | Predict: loss curve shape, accuracy, what happens without zeroing grads | [ ] |
| 5 | 45 min | Claude generates; review, run | [x] |
| 6 | 30 min | Experiments I design: LR too high/low, no hidden layer vs. one, ReLU vs. tanh | [x] |
| 7 | 30 min | Bug hunt in engine.py using only tests | [x] |
| 8 | 20 min | Trace: backprop through a 2-layer MLP with real numbers, in chat | [x] |
| 9 | 10 min | Day 1 Block 7 redo, cold: spec Value with +, *, backward() from memory; literal build must pass drill tests | [ ] |

Questions to answer (Claude logs my answers in notes.md):
- Why central differences rather than one-sided?
- Why does a too-small h make gradcheck worse?
- Why can't a network with no hidden layer separate moons?

**Done when:** every op passes gradcheck, the moons boundary plot is clean, and I found both planted bugs.

---

## Day 3: Tensor engine

Goal: the same engine on NumPy arrays, the step from toy to real.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 45 min | Derive in chat: matmul gradients with shapes; broadcasting gradients | [ ] |
| 2 | 40 min | Dictate tensor.md: elementwise ops, matmul, sum/mean with axis, reshape, transpose, iterative topo sort | [ ] |
| 3 | 15 min | Predict gradient shapes: (32,10) + (10,) bias; (32,784) @ (784,128) | [ ] |
| 4 | 60 min | Claude generates; review with focus on broadcasting code | [ ] |
| 5 | 40 min | Choose tests: gradcheck every op, 3+ broadcasting cases, a deep graph | [ ] |
| 6 | 20 min | Break it (below) | [ ] |
| 7 | 20 min | Spec from memory: matmul backward with the shape argument; literal build must pass gradcheck | [ ] |

tensor.md must explain (my words):
- For C = A @ B: why ∂L/∂A = ∂L/∂C @ Bᵀ and ∂L/∂B = Aᵀ @ ∂L/∂C. Check by matching shapes.
- How a broadcast input's gradient gets summed back to its original shape, and over which axes.
- Why the topological sort must be iterative, not recursive.

Break it (predict first):
- Remove the broadcast un-summing. Crash or silently wrong answer?
- Swap the transpose in the matmul backward.
- Build a 5,000-node chain with a recursive sort.

**Done when:** all gradchecks pass, including broadcasting.

---

## Day 4: Softmax cross-entropy and MNIST

Goal: a real model, trained end to end on my own engine.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 40 min | Derive in chat: log-softmax with max subtraction; CE gradient = softmax − one_hot | [ ] |
| 2 | 30 min | Dictate loss.md (fused log-softmax + CE) and train_mnist.md (minibatches, init, SGD/momentum, eval) | [ ] |
| 3 | 15 min | Predict: initial loss for 10 classes, final accuracy, epochs needed | [ ] |
| 4 | 60 min | Claude generates; review, train | [ ] |
| 5 | 30 min | Break it (below) | [ ] |
| 6 | 30 min | Experiments I design: batch size, LR, width, init scale | [ ] |
| 7 | 30 min | Spec from memory: full scalar engine in one message, 15 min, no notes; literal build must pass all Day 1 + Day 2 tests first try | [ ] |
| 8 | 20 min | Quiz: 10 questions covering all 4 days | [ ] |

Questions to answer:
- Why the starting loss should be about ln(10) ≈ 2.30, and what it means if it isn't.
- Why fusing softmax and cross-entropy is more stable than computing them separately.
- What a bad init scale does to the first few hundred steps.

Break it (predict first):
- Remove the max subtraction and feed in logits around 1000.
- Compute softmax and log separately and compare with the fused version.
- Initialize all weights to zero.

**Done when:** about 97% test accuracy, starting loss ≈ 2.30, and the spec-from-memory build passes first try.

---

## Stage 1 exit check

- [ ] Explain +=, reverse topological order, and the grad of 1 at the output without notes.
- [ ] Derive matmul and broadcasting gradients with shapes.
- [ ] Derive why the cross-entropy gradient is softmax − one_hot.
- [ ] Find planted bugs from tests alone.
- [ ] Spec the scalar engine from memory so a literal build passes every test first try.
- [ ] In a blind review, catch a bug in generated code by reading before running.
