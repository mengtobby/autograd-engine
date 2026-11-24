"""Neuron, Layer, MLP. Implements specs/nn.md."""

from __future__ import annotations

import random
from collections.abc import Sequence

from engine import Value

ACTIVATIONS = ("tanh", "relu")
Input = Value | float


class Neuron:
    def __init__(self, n_in: int, activation: str | None = "tanh") -> None:
        # Random in [-1, 1]: breaks symmetry, both signs, small (no saturation)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation  # None = linear (output layer)

    def __call__(self, x: Sequence[Input]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        total = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "tanh":
            return total.tanh()
        if self.activation == "relu":
            return total.relu()
        return total

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int, activation: str | None = "tanh") -> None:
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, x: Sequence[Input]) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in: int, sizes: Sequence[int], activation: str = "tanh") -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        dims = [n_in, *sizes]
        last = len(sizes) - 1
        # Hidden layers use the activation; the output layer is linear.
        self.layers = [
            Layer(dims[i], dims[i + 1], activation if i < last else None)
            for i in range(len(sizes))
        ]

    def __call__(self, x: Sequence[Input]) -> Value | list[Value]:
        out: Value | list[Value] = list(x)
        for layer in self.layers:
            out = layer(out if isinstance(out, list) else [out])
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
