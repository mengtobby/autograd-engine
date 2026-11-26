"""Fused softmax + cross-entropy. Implements specs/loss.md."""

from __future__ import annotations

import numpy as np

from tensor import Tensor


def cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean cross-entropy over a batch. logits (N, classes), labels one-hot (N, classes)."""
    one_hot = np.asarray(labels, dtype=np.float64)
    if logits.data.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, classes), got shape {logits.shape}")
    if one_hot.shape != logits.shape:
        raise ValueError(f"labels shape {one_hot.shape} must match logits shape {logits.shape}")
    n = logits.shape[0]

    # 1. Max subtraction: same p, biggest exponent is 0
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    # 2. Softmax (kept for the backward)
    e = np.exp(z)
    total = e.sum(axis=1, keepdims=True)
    p = e / total
    # 3. Cross-entropy via the rewrite -z_y + log Σ e^z: never log a p that rounded to 0
    rows = -(z * one_hot).sum(axis=1) + np.log(total[:, 0])
    # 4. Mean over the N rows
    out = Tensor(rows.mean(), (logits,), "cross_entropy")

    def _backward() -> None:
        logits.grad += (p - one_hot) / n * out.grad

    out._backward = _backward
    return out
