"""Numerical gradient check. Implements specs/gradcheck.md."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from engine import Value


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
