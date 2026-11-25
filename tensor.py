"""NumPy Tensor autograd engine. Implements specs/tensor.md."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

Number = int | float


def _as_tensor(x: Tensor | Number | np.ndarray) -> Tensor:
    """Wrap plain numbers and arrays in a leaf Tensor; pass Tensors through."""
    return x if isinstance(x, Tensor) else Tensor(x)


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a gradient over the broadcast copies, back to the input's shape."""
    # 1. Extra dimensions: sum away the extra leading ones
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. Stretched dimensions: size 1 in the input but bigger in grad
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __array_ufunc__ = None  # make `ndarray + Tensor` call Tensor.__radd__

    def __init__(self, data: object, _prev: Iterable[Tensor] = (), _op: str = "") -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)  # same shape as data
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None  # leaf: no parents, nothing to do

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    # ----- elementwise ops (broadcast) -----

    def __add__(self, other: Tensor | Number | np.ndarray) -> Tensor:
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Number | np.ndarray) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | Number | np.ndarray) -> Tensor:
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Number | np.ndarray) -> Tensor:
        return self * other

    def __pow__(self, n: Number) -> Tensor:
        if isinstance(n, Tensor) or not isinstance(n, (int, float)):
            raise TypeError("exponent must be a plain int or float")
        out = Tensor(self.data ** n, (self,), "**n")

        def _backward() -> None:
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out = Tensor(np.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ----- derived ops: no new backward rules -----

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | Number | np.ndarray) -> Tensor:
        return self + (-_as_tensor(other))

    def __rsub__(self, other: Number | np.ndarray) -> Tensor:
        return _as_tensor(other) + (-self)

    def __truediv__(self, other: Tensor | Number | np.ndarray) -> Tensor:
        return self * _as_tensor(other) ** -1

    def __rtruediv__(self, other: Number | np.ndarray) -> Tensor:
        return _as_tensor(other) * self ** -1

    # ----- matmul -----

    def __matmul__(self, other: Tensor | np.ndarray) -> Tensor:
        other = _as_tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(f"matmul is 2-D only, got {self.shape} @ {other.shape}")
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            self.grad += out.grad @ other.data.T  # A.grad = G @ Bᵀ
            other.grad += self.data.T @ out.grad  # B.grad = Aᵀ @ G

        out._backward = _backward
        return out

    # ----- reductions and shape ops -----

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # put the summed axes back as size 1
            self.grad += np.broadcast_to(g, self.shape)  # copy G back

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        total = self.sum(axis=axis, keepdims=keepdims)
        n = self.data.size // max(total.data.size, 1)  # entries averaged per output
        return total * (1.0 / n)  # sum ÷ n, gradient for free

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        out = Tensor(self.data.T, (self,), "T")

        def _backward() -> None:
            self.grad += out.grad.T

        out._backward = _backward
        return out

    # ----- backward pass: Build (iterative) → Seed → Walk back -----

    def backward(self) -> None:
        if self.data.size != 1:
            raise ValueError(f"backward needs a single-number loss, got shape {self.shape}")

        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                topo.append(node)  # all its parents are already in topo
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))  # come back to it after its parents
            for parent in node._prev:
                if parent not in visited:
                    stack.append((parent, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
