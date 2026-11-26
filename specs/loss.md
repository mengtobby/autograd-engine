# Cross-entropy loss (fused softmax + CE)

## Purpose
MNIST has 10 classes, so the network outputs 10 scores (logits) per image.
This turns a batch of logits into one number, the loss, that is small when
the model puts high probability on the correct digit.

## Interface
`cross_entropy(logits, labels) -> Tensor`
- `logits`: Tensor, shape (N, 10). One row of scores per image.
- `labels`: one-hot rows, shape (N, 10). A plain array, not a Tensor.
- Returns a single-number Tensor: the **mean** loss over the N images.

**Fused:** one new op, so the whole loss is **one** node with a hand-written
backward. It is not built from exp/sum/log nodes.

## Forward (per row z, correct class y)
1. **Max subtraction:** z − max(z). This doesn't change p, because top and
   bottom both get multiplied by e^(−c). It does stop overflow: the biggest
   exponent becomes 0, so the biggest e^ is 1.
2. **Softmax:** p_i = e^(z_i) / Σ_j e^(z_j). e^ makes every score positive and
   keeps the order; dividing by the total makes the 10 p's add to 1.
3. **Cross-entropy:** L_row = −log(p_y). Certain and right gives 0. Confident
   and wrong is huge.
   **Computed with my rewrite, not as log of p:** L_row = −z_y + log Σ_j e^(z_j)
   (on the max-subtracted z). Example: z = [0, −800] with the second class
   correct. Softmax first gives p_y = e^(−800) / (1 + e^(−800)), which rounds
   to 0 (float64 can't go below about e^(−709)), and −log(0) = inf. The rewrite
   gives 800 + log(1 + ~0) = 800, which is finite and correct. This is why
   fused is more stable: we never take the log of a number that rounded to 0.
   p is still computed for the backward.
4. **Mean** over the N rows.

## Backward
Derived in chat: L = −z_y + log Σ_j e^(z_j).
- ∂(−z_y)/∂z_k = −1 if k = y, else 0, which is −one_hot.
- ∂ log(S)/∂z_k = (1/S)·e^(z_k) = p_k.
- So per row, ∂L_row/∂z = p − one_hot.

The mean divides by N, so each row moves the batch loss by 1/N of that:

`logits.grad += (p − one_hot) / N * out.grad`, shape (N, 10), the same as logits.

The correct logit gets pushed up, and each wrong one gets pushed down by its
probability. Each row's gradient adds up to 0.

## Sanity check
At the start all logits are about equal, so p ≈ 0.1 everywhere and the loss
is ≈ ln(10) ≈ 2.30. At that point the gradient row is −0.9 for the correct
digit and +0.1 for the other nine, all ÷ N.

## Edge cases
- `logits` not 2-D, or `labels` not the same shape as `logits`: raise
  ValueError.
- Large logits (around 1000): must give a finite loss and gradient. This is
  what max subtraction is for.

## Tests I want
- Gradcheck the fused op (tensor_gradcheck) on random (4, 10) logits.
- Matches PyTorch (`torch.nn.functional.cross_entropy`): loss and gradient.
- All-equal logits give a loss of ln(10) ≈ 2.30, and the gradient row is
  (0.1 − one_hot) / N.
- Logits around 1000 give a finite loss and gradient (no nan or inf).
- Wrong shapes raise ValueError.
