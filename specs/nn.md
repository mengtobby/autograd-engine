# Neural network: Neuron, Layer, MLP

## Purpose
The calculation that learns. Built from Values, so `loss.backward()` reaches
every weight and bias and training can read their `.grad`.

## Structure
```
MLP    = a stack of Layers         (the whole network)
Layer  = a row of Neurons          (all looking at the same inputs)
Neuron = the basic unit
```

## Interface
- `Neuron(n_in)`: called as `neuron(x)` with a list of `n_in` inputs → one Value.
- `Layer(n_in, n_out)`: `n_out` Neurons, each with `n_in` inputs; `layer(x)`
  → list of `n_out` Values (a single Value if `n_out == 1`).
- `MLP(n_in, sizes)`: e.g. `MLP(2, [4, 1])`; layer k's input count is the
  previous layer's size. `mlp(x)` runs the layers in order.
- `parameters()` on all three: a flat list of every weight and bias Value.
  The training loop uses it to update and to reset gradients.

## Forward (Neuron)
    out = activation( w₁·x₁ + … + wₙ·xₙ + b )

- **Weights and bias are Values**: training needs `w.grad`
  (`w = w − lr × w.grad`). Only Values get gradients; plain numbers are
  constants. If they were floats, backward would pass right by them.
- **Inputs can be plain floats**: the engine wraps them automatically when
  multiplied by a weight. They become leaves whose grads nobody reads.

## Parameter count
A neuron has one weight per input it receives, plus a bias. A layer's inputs
are the previous layer's outputs.
`MLP(2, [4, 1])`: hidden 4 × (2 + 1) = 12, output 1 × (4 + 1) = 5 → **17**.

## Initialization
Weights and biases start **random, uniform between −1 and 1**:
- **Random** (breaks symmetry): if all start the same (all 0 or all 1),
  every neuron in a layer gives the same output, gets the same gradients and
  stays identical forever.
- **Both signs**: a weight's sign says whether an input pushes the neuron up
  or down; 0 to 1 would start every neuron pushing the same way.
- **Small** (avoids saturation): a big weight puts tanh's input near ±50, so
  out ≈ ±1 and the local gradient 1 − out² ≈ 0. The weights' gradients are
  zeroed, so they never change; the neuron is stuck.

## Edge cases
- Input list length must equal `n_in` (raise ValueError otherwise).
- **Hidden-layer activation** is a setting on the MLP:
  `MLP(n_in, sizes, activation="tanh")`, with `"relu"` as the other option
  (any other value raises ValueError). Lets the Block 6 ReLU vs. tanh
  experiment change one setting.
- **Output layer is linear** (raw weighted sum, no activation). With tanh, a
  confident prediction lands in the saturated zone and its gradient goes to
  ≈ 0. A linear output can be any number; the loss decides how to judge it.

## Tests I want
- `MLP(2, [4, 1])` has exactly 17 parameters, all Values.
- Initial weights are all within [−1, 1] and not all equal.
- Every parameter gets a gradient: gradcheck a small MLP's loss with respect
  to its weights.
- Wrong input length raises ValueError.
- The output layer applies no activation: its output equals its weighted sum.
- `activation="relu"` uses relu in hidden layers; an unknown name raises ValueError.
