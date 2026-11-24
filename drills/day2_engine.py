"""Day 2 Block 9 drill: literal build of my recalled spec (after review).
MINE = from my answers. GAP = what my spec didn't say, left out."""

from __future__ import annotations


class Value:
    def __init__(self, data: float, _prev: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = data  # MINE: raw float
        self.grad = 0  # MINE: starts at 0
        self._prev = _prev  # MINE: parents
        self._op = _op  # MINE: operation
        self._backward = lambda: None  # MINE (round 2): walk back calls backward

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)  # MINE: wrap
        out = Value(self.data + other.data, (self, other), "+")  # MINE

        def _backward() -> None:
            self.grad += out.grad  # MINE: a.grad += out.grad
            other.grad += out.grad  # MINE: b.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)  # MINE: wrap
        out = Value(self.data * other.data, (self, other), "*")  # MINE

        def _backward() -> None:
            self.grad += other.data * out.grad  # MINE: a.grad += b.data * out.grad
            other.grad += self.data * out.grad  # MINE: b.grad += a.data * out.grad

        out._backward = _backward
        return out

    # GAP: no __radd__ / __rmul__ (named _as_value instead), so `2 * a` fails

    def backward(self) -> None:
        topo: list[Value] = []

        def build(node: Value) -> None:
            # MINE: "built by visiting the parents" (parents first, then the node)
            # GAP: no visited check, so a node reached twice is listed twice
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)  # MINE: build
        self.grad = 1  # MINE: seed
        for node in reversed(topo):  # MINE: walk back
            node._backward()
