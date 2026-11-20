"""Day 1 tests. Each one comes from a "Tests I want" section in specs/ or
the PLAN.md Block 5 list. PyTorch is the reference for gradients."""

import pytest
import torch

from engine import Value


def t(x: float) -> torch.Tensor:
    """A float64 PyTorch leaf that tracks gradients."""
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


# ----- value.md -----

def test_leaf_fields() -> None:
    a = Value(2.0)
    assert (a.data, a.grad, a._prev, a._op) == (2.0, 0, (), "")
    a._backward()
    assert a.grad == 0  # a leaf's _backward does nothing


def test_op_records_parents_and_label() -> None:
    a, b = Value(2.0), Value(-3.0)
    e = a * b
    assert e._prev == (a, b)
    assert e._op == "*"


# ----- add.md -----

def test_a_plus_a() -> None:
    a = Value(3.0)
    (a + a).backward()
    assert a.grad == 2


def test_add_chain_vs_torch() -> None:
    a, b, c = Value(1.5), Value(-2.0), Value(4.0)
    (a + b + c).backward()
    ta, tb, tc = t(1.5), t(-2.0), t(4.0)
    (ta + tb + tc).backward()
    assert [a.grad, b.grad, c.grad] == pytest.approx([ta.grad.item(), tb.grad.item(), tc.grad.item()])


# ----- ops.md -----

def test_a_times_a() -> None:
    a = Value(3.0)
    (a * a).backward()
    assert a.grad == 6


@pytest.mark.parametrize("n, expected", [(2, 6), (3, 27)])
def test_pow(n: int, expected: float) -> None:
    a = Value(3.0)
    (a ** n).backward()
    assert a.grad == pytest.approx(expected)


def test_pow_rejects_value_exponent() -> None:
    with pytest.raises(TypeError):
        Value(3.0) ** Value(2.0)


@pytest.mark.parametrize("expr", [lambda a: 2 * a, lambda a: a * 2])
def test_plain_number_either_side(expr) -> None:
    a = Value(3.0)
    out = expr(a)
    out.backward()
    assert (out.data, a.grad) == (6, 2)


@pytest.mark.parametrize("x, expected", [(2.0, 1), (-2.0, 0), (0.0, 0)])
def test_relu(x: float, expected: float) -> None:
    a = Value(x)
    a.relu().backward()
    assert a.grad == expected


def test_tanh_exp_chain_vs_torch() -> None:
    a = Value(0.7)
    L = (a * 2).tanh().exp() * a
    L.backward()
    ta = t(0.7)
    tL = (ta * 2).tanh().exp() * ta
    tL.backward()
    assert L.data == pytest.approx(tL.item())
    assert a.grad == pytest.approx(ta.grad.item())


# ----- backward.md -----

def test_seed_is_one() -> None:
    L = Value(5.0)
    L.backward()
    assert L.grad == 1


def test_worked_example() -> None:
    w, x = Value(3.0), Value(2.0)
    L = (w * x + (-4)) ** 2
    L.backward()
    assert (w.grad, x.grad) == (8, 12)


def test_shared_node_waits_for_every_user() -> None:
    w = Value(1.0)
    x = w * 2
    L = x * 3 + x * 4
    L.backward()
    assert (x.grad, w.grad) == (7, 14)


@pytest.mark.parametrize("op", [
    lambda a, b: -a,
    lambda a, b: a - b,
    lambda a, b: 3 - a,
    lambda a, b: a / b,
    lambda a, b: 3 / a,
])
def test_derived_ops_vs_torch(op) -> None:
    a, b = Value(6.0), Value(2.0)
    out = op(a, b)
    out.backward()
    ta, tb = t(6.0), t(2.0)
    tout = op(ta, tb)
    tout.backward()
    assert out.data == pytest.approx(tout.item())
    assert a.grad == pytest.approx(ta.grad.item() if ta.grad is not None else 0)
    assert b.grad == pytest.approx(tb.grad.item() if tb.grad is not None else 0)


# ----- Block 3: my hand trace, checked two ways -----

def test_block3_trace_matches_my_prediction() -> None:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    e = a * b
    d = e + c
    L = d * d
    L.backward()
    assert (e.data, d.data, L.data) == (-6, 4, 16)
    assert (L.grad, d.grad, e.grad, c.grad, a.grad, b.grad) == (1, 8, 8, 8, -24, 16)


def test_block3_trace_vs_torch() -> None:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c
    (d * d).backward()
    ta, tb, tc = t(2.0), t(-3.0), t(10.0)
    td = ta * tb + tc
    (td * td).backward()
    assert [a.grad, b.grad, c.grad] == pytest.approx([ta.grad.item(), tb.grad.item(), tc.grad.item()])
