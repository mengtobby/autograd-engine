"""Tests from specs/gradcheck.md "Tests I want"."""

import pytest

from engine import Value
from gradcheck import gradcheck

# (name, f, inputs): every op, away from kinks
OPS = [
    ("add", lambda v: v[0] + v[1], [1.5, -2.0]),
    ("mul", lambda v: v[0] * v[1], [1.5, -2.0]),
    ("pow 2", lambda v: v[0] ** 2, [1.3]),
    ("pow 3", lambda v: v[0] ** 3, [1.3]),
    ("pow -1", lambda v: v[0] ** -1, [1.3]),
    ("pow 0.5", lambda v: v[0] ** 0.5, [1.3]),
    ("tanh", lambda v: v[0].tanh(), [0.4]),
    ("exp", lambda v: v[0].exp(), [0.4]),
    ("relu +0.5", lambda v: v[0].relu(), [0.5]),
    ("relu -0.5", lambda v: v[0].relu(), [-0.5]),
    ("relu +2", lambda v: v[0].relu(), [2.0]),
    ("relu -2", lambda v: v[0].relu(), [-2.0]),
    ("neg", lambda v: -v[0], [1.5]),
    ("sub", lambda v: v[0] - v[1], [1.5, -2.0]),
    ("div", lambda v: v[0] / v[1], [1.5, -2.0]),
]


@pytest.mark.parametrize("name, f, inputs", OPS, ids=[o[0] for o in OPS])
def test_every_op_passes(name, f, inputs) -> None:
    gradcheck(f, inputs)


def test_block3_graph() -> None:
    def f(v):
        d = v[0] * v[1] + v[2]
        return d * d

    gradcheck(f, [2.0, -3.0, 10.0])


def test_tanh_exp_chain() -> None:
    gradcheck(lambda v: (v[0] * 2).tanh().exp() * v[0], [0.7])


# ----- self-tests: the checker must be able to fail -----

def _broken_square(x: Value) -> Value:
    """x ** 2 with a wrong backward: forgets the factor of 2."""
    out = Value(x.data ** 2, (x,), "broken")

    def _backward() -> None:
        x.grad += x.data * out.grad  # should be 2 * x.data

    out._backward = _backward
    return out


def test_catches_a_wrong_gradient() -> None:
    with pytest.raises(AssertionError, match="input 0"):
        gradcheck(lambda v: _broken_square(v[0]), [1.3])


def test_relu_at_kink_fails() -> None:
    with pytest.raises(AssertionError):
        gradcheck(lambda v: v[0].relu(), [0.0])
