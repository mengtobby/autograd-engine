"""Tests from specs/tensor.md "Tests I want"."""

import numpy as np
import pytest
import torch

from gradcheck import tensor_gradcheck
from tensor import Tensor

rng = np.random.default_rng(0)


def rand(*shape: int) -> np.ndarray:
    return rng.normal(size=shape)


def away_from_zero(*shape: int) -> np.ndarray:
    """Random values with |x| >= 0.5: safe for relu's kink and for division."""
    x = rng.uniform(0.5, 2.0, size=shape)
    return x * rng.choice([-1.0, 1.0], size=shape)


def scored(fn, out_shape: tuple[int, ...]):
    """Turn an op into a single-number loss: sum(op(...) * R) with fixed random
    R, so G = out.grad is a varied array, not all ones."""
    R = rand(*out_shape)
    return lambda ts: (fn(ts) * R).sum()


# (name, op, input arrays, output shape)
OPS = [
    ("add", lambda t: t[0] + t[1], [rand(3, 2), rand(3, 2)], (3, 2)),
    ("mul", lambda t: t[0] * t[1], [rand(3, 2), rand(3, 2)], (3, 2)),
    ("pow 2", lambda t: t[0] ** 2, [rand(3, 2)], (3, 2)),
    ("pow 3", lambda t: t[0] ** 3, [rand(3, 2)], (3, 2)),
    ("pow -1", lambda t: t[0] ** -1, [away_from_zero(3, 2)], (3, 2)),
    ("pow 0.5", lambda t: t[0] ** 0.5, [rng.uniform(0.5, 2, (3, 2))], (3, 2)),
    ("tanh", lambda t: t[0].tanh(), [rand(3, 2)], (3, 2)),
    ("exp", lambda t: t[0].exp(), [rand(3, 2)], (3, 2)),
    ("relu", lambda t: t[0].relu(), [away_from_zero(3, 2)], (3, 2)),
    ("neg", lambda t: -t[0], [rand(3, 2)], (3, 2)),
    ("sub", lambda t: t[0] - t[1], [rand(3, 2), rand(3, 2)], (3, 2)),
    ("div", lambda t: t[0] / t[1], [rand(3, 2), away_from_zero(3, 2)], (3, 2)),
    ("matmul", lambda t: t[0] @ t[1], [rand(2, 3), rand(3, 4)], (2, 4)),
    ("sum all", lambda t: t[0].sum(), [rand(3, 2)], ()),
    ("sum axis 0", lambda t: t[0].sum(axis=0), [rand(3, 2)], (2,)),
    ("sum axis 1 keepdims", lambda t: t[0].sum(axis=1, keepdims=True), [rand(3, 2)], (3, 1)),
    ("mean all", lambda t: t[0].mean(), [rand(3, 2)], ()),
    ("mean axis 1", lambda t: t[0].mean(axis=1), [rand(3, 2)], (3,)),
    ("reshape", lambda t: t[0].reshape(3, 2), [rand(2, 3)], (3, 2)),
    ("transpose", lambda t: t[0].T, [rand(2, 3)], (3, 2)),
    # broadcasting
    ("bcast row (3,2)+(2,)", lambda t: t[0] + t[1], [rand(3, 2), rand(2)], (3, 2)),
    ("bcast col (3,2)+(3,1)", lambda t: t[0] + t[1], [rand(3, 2), rand(3, 1)], (3, 2)),
    ("bcast number (3,2)+5", lambda t: t[0] + 5.0, [rand(3, 2)], (3, 2)),
    ("bcast mul (3,2)*(2,)", lambda t: t[0] * t[1], [rand(3, 2), rand(2)], (3, 2)),
    ("bcast bias layer", lambda t: (t[0] @ t[1] + t[2]).tanh(), [rand(4, 3), rand(3, 2), rand(2)], (4, 2)),
]


@pytest.mark.parametrize("name, op, arrays, out_shape", OPS, ids=[o[0] for o in OPS])
def test_gradcheck_every_op(name, op, arrays, out_shape) -> None:
    tensor_gradcheck(scored(op, out_shape), arrays)


# ----- worked examples from Block 1 -----

G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_row_bias_sums_down_rows() -> None:
    X, b = Tensor(np.ones((3, 2))), Tensor([10.0, 20.0])
    ((X + b) * G).sum().backward()  # makes out.grad = G
    assert b.grad.shape == (2,)
    assert b.grad.tolist() == [9.0, 12.0]


def test_column_sums_across_columns() -> None:
    X, c = Tensor(np.ones((3, 2))), Tensor([[100.0], [200.0], [300.0]])
    ((X + c) * G).sum().backward()
    assert c.grad.shape == (3, 1)
    assert c.grad.tolist() == [[3.0], [7.0], [11.0]]


def test_sum_backward_is_all_ones() -> None:
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = A.sum()
    out.backward()
    assert out.data == 21
    assert A.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


# ----- Block 3 predictions -----

def test_block3_shapes() -> None:
    X, b = Tensor(rand(32, 10)), Tensor(rand(10))
    (X + b).sum().backward()
    assert b.grad.shape == (10,)

    X, W = Tensor(rand(32, 784)), Tensor(rand(784, 128))
    H = X @ W
    assert H.shape == (32, 128)
    (H * rand(32, 128)).sum().backward()
    assert W.grad.shape == (784, 128)
    assert X.grad.shape == (32, 784)


# ----- PyTorch, deep graph, errors -----

def test_matmul_vs_torch() -> None:
    a, b, r = rand(2, 3), rand(3, 4), rand(2, 4)
    A, B = Tensor(a), Tensor(b)
    ((A @ B) * r).sum().backward()
    ta = torch.tensor(a, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    ((ta @ tb) * torch.tensor(r)).sum().backward()
    np.testing.assert_allclose(A.grad, ta.grad.numpy())
    np.testing.assert_allclose(B.grad, tb.grad.numpy())


def test_deep_chain_5000() -> None:
    x = Tensor(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.001
    y.backward()  # recursion would hit RecursionError
    assert x.grad == pytest.approx(1.001 ** 5000)


def test_backward_needs_single_number() -> None:
    with pytest.raises(ValueError):
        Tensor(rand(3, 2)).backward()


def test_matmul_is_2d_only() -> None:
    with pytest.raises(ValueError):
        Tensor(rand(3)) @ Tensor(rand(3, 2))


def test_numpy_array_on_the_left() -> None:
    A = Tensor(rand(3, 2))
    out = rand(3, 2) + A
    assert isinstance(out, Tensor)
