# Stage 1 plan: autograd from scratch

Four days: scalar engine → gradcheck + MLP → tensor engine → softmax CE + MNIST.
Originally scheduled Nov 17–20; dates slide, the order doesn't.

## The loop for every piece

1. **Spec** in `specs/`
2. **Derive** on paper (photo it or summarize in the spec)
3. **Predict** in `notes.md`
4. **Generate**: "Implement specs/<file>.md in <module>.py"
5. **Review**: annotate every line; ask about anything unclear
6. **Verify**: run your tests
7. **Break**: one deliberate change, predict it, run it
8. **Commit** with what you learned in the message

---

## Day 1: Scalar engine

Goal: a `Value` class whose gradients match PyTorch.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 45 min | On paper: local gradients for +, ×, **n, tanh, exp, ReLU | [ ] |
| 2 | 30 min | Specs: value.md (fields), ops.md (all ops), backward.md | [ ] |
| 3 | 15 min | Predict: hand-compute every gradient for a=2, b=-3, c=10, d=a*b+c, L=d*d | [ ] |
| 4 | 45 min | Generate, review line by line, answer the AI's 2 questions | [ ] |
| 5 | 30 min | Tests: that expression, a + a, a * a, a tanh/exp chain, all vs. PyTorch | [ ] |
| 6 | 20 min | Break it (below) | [ ] |
| 7 | 30 min | Hand drill: Value with + and × plus backward() in drills/ | [ ] |

backward.md must answer:
- Why the output's grad starts at 1.
- Why topological order, and why reversed.
- Why += rather than =.
- How subtraction, division and negation come from existing ops.

Break it:
- `=` instead of `+=`: which test fails?
- Skip the topological sort: when does it still give the right answer by luck?
- Don't set the output grad to 1: what do all the gradients become?

**Done when:** all tests pass and your hand predictions match.

---

## Day 2: Gradcheck and MLP

Goal: prove the engine is correct in general, then train a network with it.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 30 min | Spec gradcheck.md: central differences, choose h and tolerance, explain the choice | [ ] |
| 2 | 30 min | Spec nn.md: Neuron, Layer, MLP, parameters(), weight initialization | [ ] |
| 3 | 20 min | Spec train_moons.md: forward, loss (hinge or MSE), zero grads, backward, SGD step | [ ] |
| 4 | 15 min | Predict: loss curve shape, accuracy, and what happens without zeroing grads | [ ] |
| 5 | 45 min | Generate, review, run | [ ] |
| 6 | 30 min | Experiments: LR too high/low, no hidden layer vs. one, ReLU vs. tanh | [ ] |
| 7 | 30 min | Bug hunt in engine.py using only your tests | [ ] |
| 8 | 20 min | Hand drill: backprop for a 2-layer MLP on paper with real numbers | [ ] |

Questions to answer in notes:
- Why central differences rather than one-sided?
- Why does a too-small h make gradcheck worse?
- Why can't a network with no hidden layer separate moons?

**Done when:** every op passes gradcheck, the moons boundary plot is clean, and you found both planted bugs.

---

## Day 3: Tensor engine

Goal: the same engine on NumPy arrays, the step from toy to real.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 45 min | On paper: matmul gradients with shapes; broadcasting gradients | [ ] |
| 2 | 40 min | Spec tensor.md: elementwise ops, matmul, sum/mean with axis, reshape, transpose, iterative topological sort | [ ] |
| 3 | 15 min | Predict gradient shapes: (32,10) + (10,) bias; (32,784) @ (784,128) | [ ] |
| 4 | 60 min | Generate, review (focus on broadcasting code) | [ ] |
| 5 | 40 min | Tests: gradcheck every op, at least 3 broadcasting cases, a deep graph | [ ] |
| 6 | 20 min | Break it | [ ] |
| 7 | 20 min | Hand drill: derive and write matmul backward from memory | [ ] |

Spec must explain:
- For C = A @ B: why ∂L/∂A = ∂L/∂C @ Bᵀ and ∂L/∂B = Aᵀ @ ∂L/∂C. Check by matching shapes.
- How a broadcast input's gradient gets summed back to its original shape, and which axes.
- Why the topological sort must be iterative, not recursive.

Break it:
- Remove the broadcast un-summing. Crash or silently wrong answer?
- Swap the transpose in the matmul backward.
- Build a 5,000-node chain with a recursive sort.

**Done when:** all gradchecks pass, including broadcasting.

---

## Day 4: Softmax cross-entropy and MNIST

Goal: a real model, trained end to end on your own engine.

| Block | Time | Task | Done |
|---|---|---|---|
| 1 | 40 min | On paper: log-softmax with max subtraction; derive that the CE gradient is softmax − one_hot | [ ] |
| 2 | 30 min | Spec loss.md (fused log-softmax + CE) and train_mnist.md (minibatches, init, SGD/momentum, eval) | [ ] |
| 3 | 15 min | Predict: initial loss for 10 classes, final accuracy, epochs needed | [ ] |
| 4 | 60 min | Generate, review, train | [ ] |
| 5 | 30 min | Break it | [ ] |
| 6 | 30 min | Experiments: batch size, LR, width, init scale | [ ] |
| 7 | 30 min | Hand drill: full scalar engine from a blank file, 30 min | [ ] |
| 8 | 20 min | Quiz: 10 questions covering all 4 days | [ ] |

Questions to answer:
- Why the starting loss should be about ln(10) ≈ 2.30, and what it means if it isn't.
- Why fusing softmax and cross-entropy is more stable than computing them separately.
- What a bad init scale does to the first few hundred steps.

Break it:
- Remove the max subtraction and feed in logits around 1000.
- Compute softmax and log separately and compare against the fused version.
- Initialize all weights to zero.

**Done when:** about 97% test accuracy, starting loss ≈ 2.30, and the hand drill under 30 minutes.

---

## Stage 1 exit check

- [ ] Explain +=, reverse topological order, and the grad of 1 at the output without notes.
- [ ] Derive matmul and broadcasting gradients with shapes.
- [ ] Derive why the cross-entropy gradient is softmax − one_hot.
- [ ] Find planted bugs from tests alone.
- [ ] Write the scalar engine by hand in 30 minutes.
