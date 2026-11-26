# Train on MNIST

## Purpose
Train a Tensor MLP (specs/tensor.md) to read handwritten digits: 28×28
grayscale images (784 pixels), labels 0–9. The target is about 97% test
accuracy, using my own engine and fused loss (specs/loss.md).

## Data
- 60,000 training images and 10,000 test images.
- **Pixels ÷ 255**, so inputs fall in 0–1. Raw 0–255 would make huge sums
  (tanh saturates) and huge weight gradients (W.grad = Xᵀ @ G scales with x).
- Labels become one-hot rows (N, 10), as loss.md expects.
- Source: `sklearn.datasets.fetch_openml("mnist_784")`, cached in
  data/ (gitignored). Standard split: first 60,000 for training, last 10,000
  for testing.

## Model
`784 → 128 (tanh) → 10 (no activation)`.
- One hidden layer of 128 is enough for ~97%. With no hidden layer it tops
  out around 92%, because a linear model draws straight boundaries.
- The output layer is linear: the logits go straight into the fused
  cross-entropy, which does its own softmax.
- Each layer is `X @ W + b`. W is (n_in, n_out) and b is (n_out,), added by
  broadcasting, so b.grad sums over the batch (Day 3).
- Parameters: 784·128 + 128 + 128·10 + 10 = 101,770.

## Init
Weights scaled by **1/√n_in**: 1/√784 ≈ 0.036 for the first layer and
1/√128 ≈ 0.088 for the output layer.

**Why:** a neuron's sum grows like √(number of inputs) × weight size, because
random signs partly cancel. With Day 2's uniform(−1, 1) and about 100 pixels
on, the sum is ≈ √100 × 0.58 ≈ 6. tanh(6) ≈ 1, so 1 − tanh² ≈ 0 and the
gradient is zeroed (saturation). Scaling by 1/√n_in keeps the sum around 1,
where tanh still has slope.

- Exact form: W = uniform(−1, 1) / √n_in, which is Day 2's init
  scaled down.
- Biases: start at 0. The random weights already break symmetry.
- Seed: fixed (`np.random.default_rng(0)`), so experiments
  compare fairly.

## Training
- Minibatches of **B = 64**: 60,000 / 64 ≈ 938 updates per epoch.
- **Shuffle every epoch**, so each batch gets a fair mix of digits and a
  sorted file can't make the model drift toward the last digit it saw.
- **Plain SGD**, lr = 0.1. Each step: forward + loss → backward → update
  `p.data -= lr * p.grad` → zero grads (same as moons).
- **10 epochs.**
- If the last batch of an epoch isn't full: use it anyway (it is
  smaller). The loss is a mean, so its scale doesn't change.

## Eval
- **Test accuracy every epoch**, on the 10,000 test images the model never
  trains on, so we measure generalizing, not memorizing. It's a forward pass
  only (no backward), in chunks.
- Prediction = argmax of the logits.

## Logging
- Print the **first batch's loss before any update**. It should be ≈ 2.30
  (ln 10). If it isn't, stop: init or loss is wrong.
- Print the average training loss every 100 steps, plus test accuracy each
  epoch.
- Save a loss curve to mnist_loss.png.

## Tests I want
- Parameter count is 101,770.
- Starting loss on a batch with fresh init is within 0.1 of ln(10).
- Gradcheck the full model + loss on a tiny input (2 images, a hidden layer
  of 4) so it runs quickly.
- One SGD step on a batch lowers the loss on that batch.
- Short run (1 epoch on 5,000 images) beats 85% test accuracy.
- The full 10-epoch run is a script I run, not a test (too slow for pytest).
