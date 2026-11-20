"""Day 1 Block 7 drill: build of my from-memory spec, round 2.
MINE = from my answers. FILLED = gaps Claude filled in after I asked."""

from __future__ import annotations


class Value:
    def __init__(self, data: float, _prev: tuple[Value, ...] = ()) -> None:
        self.data = data  # MINE
        self.grad = 0  # FILLED: nothing collected yet (the seed of 1 is only for L)
        self._prev = _prev  # MINE
        self._backward = lambda: None  # FILLED: a leaf has no parents, so does nothing

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)  # MINE: wrap
        out = Value(self.data + other.data, (self, other))  # MINE

        def _backward() -> None:
            self.grad += out.grad  # MINE
            other.grad += out.grad  # MINE

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Value:  # FILLED: 3 + a
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)  # MINE: wrap
        out = Value(self.data * other.data, (self, other))  # MINE

        def _backward() -> None:
            self.grad += other.data * out.grad  # FILLED: a's own line
            other.grad += self.data * out.grad  # MINE (round 2: grad -> data)

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Value:  # FILLED: 2 * a
        return self * other

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:  # FILLED: step 1, parents first, then node
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = 1  # MINE (round 2): step 2, the seed
        for node in reversed(topo):  # MINE: "walking the list in reverse"
            node._backward()  # FILLED: call each node's rule
