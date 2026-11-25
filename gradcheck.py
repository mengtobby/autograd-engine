"""Numerical gradient check. Implements specs/gradcheck.md."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from engine import Value
from tensor import Tensor


def _loss_at(f: Callable[[list[Value]], Value], xs: Sequence[float]) -> float:
    """Forward pass only: build L from fresh Values and return its number."""
    return f([Value(x) for x in xs]).data


def _nudged(xs: Sequence[float], i: int, delta: float) -> list[float]:
    """A copy of xs with input i moved by delta; the others left alone."""
    return [x + delta if j == i else x for j, x in enumerate(xs)]


def gradcheck(
    f: Callable[[list[Value]], Value],
    inputs: Sequence[float],
    h: float = 1e-6,
    tol: float = 1e-6,
) -> None:
    if not inputs:
        raise ValueError("gradcheck needs at least one input")
    xs = [float(x) for x in inputs]

    # 1. Engine's answer
    leaves = [Value(x) for x in xs]
    f(leaves).backward()

    for i, leaf in enumerate(leaves):
        # 2. Central difference, nudging input i only
        estimate = (_loss_at(f, _nudged(xs, i, h)) - _loss_at(f, _nudged(xs, i, -h))) / (2 * h)

        # 3. Relative error with a noise floor
        error = abs(leaf.grad - estimate)
        if error > tol * (1 + abs(estimate)):
            raise AssertionError(
                f"input {i} (x = {xs[i]}): engine grad {leaf.grad}, "
                f"estimate {estimate}, error {error:.2e}"
            )


def tensor_gradcheck(
    f: Callable[[list[Tensor]], Tensor],
    arrays: Sequence[np.ndarray],
    h: float = 1e-6,
    tol: float = 1e-6,
) -> None:
    """Tensor version (specs/tensor.md): nudge each entry of each input one at a
    time, with the same central difference and tolerance rule as gradcheck."""
    if not arrays:
        raise ValueError("tensor_gradcheck needs at least one input")
    xs = [np.array(a, dtype=np.float64) for a in arrays]

    leaves = [Tensor(x) for x in xs]
    f(leaves).backward()

    for i, leaf in enumerate(leaves):
        for idx in np.ndindex(xs[i].shape):
            plus = [x.copy() for x in xs]
            minus = [x.copy() for x in xs]
            plus[i][idx] += h
            minus[i][idx] -= h
            l_plus = f([Tensor(x) for x in plus]).data.item()
            l_minus = f([Tensor(x) for x in minus]).data.item()
            estimate = (l_plus - l_minus) / (2 * h)
            error = abs(leaf.grad[idx] - estimate)
            if error > tol * (1 + abs(estimate)):
                raise AssertionError(
                    f"input {i} entry {idx}: engine grad {leaf.grad[idx]}, "
                    f"estimate {estimate}, error {error:.2e}"
                )
