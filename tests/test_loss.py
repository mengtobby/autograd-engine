"""Tests from specs/loss.md "Tests I want"."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gradcheck import tensor_gradcheck
from loss import cross_entropy
from tensor import Tensor

rng = np.random.default_rng(0)


def one_hot(labels: list[int], n_classes: int = 10) -> np.ndarray:
    return np.eye(n_classes)[labels]


def test_gradcheck_fused_op() -> None:
    Y = one_hot([3, 0, 9, 3])
    tensor_gradcheck(lambda ts: cross_entropy(ts[0], Y), [rng.normal(size=(4, 10))])


def test_matches_pytorch() -> None:
    z, labels = rng.normal(size=(5, 10)), [1, 4, 4, 0, 7]
    logits = Tensor(z)
    loss = cross_entropy(logits, one_hot(labels))
    loss.backward()

    tz = torch.tensor(z, requires_grad=True)
    t_loss = F.cross_entropy(tz, torch.tensor(labels))
    t_loss.backward()

    assert loss.data.item() == pytest.approx(t_loss.item())
    np.testing.assert_allclose(logits.grad, tz.grad.numpy())


def test_all_equal_logits_give_ln10() -> None:
    Y = one_hot([7, 2, 5])
    logits = Tensor(np.zeros((3, 10)))
    loss = cross_entropy(logits, Y)
    loss.backward()
    assert loss.data.item() == pytest.approx(math.log(10))
    np.testing.assert_allclose(logits.grad, (0.1 - Y) / 3)  # -0.9 / 3 for the correct digit


def test_logits_around_1000_are_finite() -> None:
    logits = Tensor(1000 + rng.normal(size=(4, 10)))
    loss = cross_entropy(logits, one_hot([0, 1, 2, 3]))
    loss.backward()
    assert np.isfinite(loss.data).all()
    assert np.isfinite(logits.grad).all()


def test_confident_and_wrong_is_finite() -> None:
    """Spec example: z = [0, -800] after max subtraction, second class correct."""
    logits = Tensor([[1000.0, 200.0]])
    loss = cross_entropy(logits, np.array([[0.0, 1.0]]))
    loss.backward()
    assert loss.data.item() == pytest.approx(800.0)
    assert np.isfinite(logits.grad).all()


def test_logits_not_2d_raise() -> None:
    with pytest.raises(ValueError):
        cross_entropy(Tensor(rng.normal(size=10)), one_hot([3])[0])


def test_label_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cross_entropy(Tensor(rng.normal(size=(4, 10))), one_hot([1, 2, 3]))
