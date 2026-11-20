"""Scalar autograd engine. Implements specs/value.md, specs/add.md,
specs/ops.md and specs/backward.md."""

from __future__ import annotations

import math
from collections.abc import Iterable

Number = int | float


def _as_value(x: Value | Number) -> Value:
    """Wrap a plain number in a leaf Value; pass Values through (ops.md)."""
    return x if isinstance(x, Value) else Value(x)


class Value:
    def __init__(self, data: Number, _prev: Iterable[Value] = (), _op: str = "") -> None:
        self.data = data
        self.grad: Number = 0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None  # leaf: no parents, nothing to do

    # ----- core ops (add.md, ops.md) -----

    def __add__(self, other: Value | Number) -> Value:
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Number) -> Value:  # 3 + a
        return self + other

    def __mul__(self, other: Value | Number) -> Value:
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Number) -> Value:  # 2 * a
        return self * other

    def __pow__(self, n: Number) -> Value:
        if isinstance(n, Value) or not isinstance(n, (int, float)):
            raise TypeError("exponent must be a plain int or float, not a Value")
        out = Value(self.data ** n, (self,), "**n")

        def _backward() -> None:
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0, (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ----- derived ops (backward.md): no new backward rules -----

    def __neg__(self) -> Value:  # -a
        return self * -1

    def __sub__(self, other: Value | Number) -> Value:  # a - b
        return self + (-_as_value(other))

    def __rsub__(self, other: Number) -> Value:  # 3 - a
        return _as_value(other) + (-self)

    def __truediv__(self, other: Value | Number) -> Value:  # a / b
        return self * _as_value(other) ** -1

    def __rtruediv__(self, other: Number) -> Value:  # 3 / a
        return _as_value(other) * self ** -1

    # ----- backward pass (backward.md) -----

    def backward(self) -> None:
        topo = []
        visited = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()
