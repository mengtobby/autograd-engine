# Train on moons

## Purpose
Train an MLP (specs/nn.md) to separate the two moons: 2D points labeled +1
(top) or −1 (bottom). A straight line can't separate them, so the network
must learn a curved boundary. We want score > 0 for +1 and score < 0 for −1.

## One training step
1. **Forward + loss**: run the MLP on every point and compute L.
2. **Backward**: `L.backward()` fills every parameter's `.grad`.
3. **Update**: for every parameter, `p.data -= lr * p.grad`.
4. **Zero grads**: set every parameter's `.grad` back to 0.

(Steps 4 → 1 → 2 → 3 is equivalent: what matters is that grads are 0 before
the next backward. The first step is fine because grads start at 0.)

**Why zero grads:** we don't want gradients computed in previous runs to
affect the training of later runs. Backward uses `+=`, so without a reset
w.grad would be 5 after step 1 and 5 + 3 = 8 after step 2, and the update
would use 8 instead of the true 3.

## Loss: hinge
Per point with label y (±1) and score s: `max(0, 1 − y·s)`, written as
`(1 − y·s).relu()` (relu is max(0, ·), so the gradient comes for free).
L = **mean** over all points.

Why hinge, not MSE: example y = +1, s = 5 (confident and correct). MSE
(s − y)² = 16, and its gradient pushes s back down to 1: it punishes being
right. Hinge = max(0, 1 − 5) = max(0, −4) = 0: correct and confident means
no error, nothing to fix. Effort goes to the misclassified points.

## Settings (PROPOSED: approve or change)
| setting | proposed | why |
|---|---|---|
| data | 100 points, `sklearn.datasets.make_moons(noise=0.1, random_state=0)`, labels mapped 0/1 → −1/+1 | small enough for a scalar engine |
| model | `MLP(2, [16, 16, 1])` (337 parameters), tanh hidden | two hidden layers to bend the boundary |
| loss over points | mean, not sum | the loss's scale (and so the right lr) doesn't depend on how many points there are |
| learning rate | 0.1 | moderate; Block 6 tries too high / too low |
| steps | 100, full batch (all 100 points every step) | 100 points is small; no minibatches needed until MNIST |
| seed | fixed `random.seed(0)` for weight init | runs are repeatable, so experiments compare fairly |

## Output
- Print step, loss and accuracy every 10 steps.
  Accuracy = fraction of points where sign(score) equals the label.
- Save `moons_boundary.png`: the points colored by label over the
  network's decision regions (score > 0 vs < 0) on a grid.
- Save `moons_loss.png`: loss vs step.

## Tests I want
- One step on a fixed tiny dataset lowers the loss.
- After step 4 every parameter's grad is 0.
- Hinge loss of (y = +1, s = 5) is 0; of (y = +1, s = −1) is 2.
- Training to the end reaches ≥ 95% accuracy on the 100 points.
