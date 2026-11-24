"""Tests from specs/train_moons.md "Tests I want"."""

import random

import pytest

from engine import Value
from nn import MLP
from train_moons import accuracy, hinge, loss_fn, train, train_step

TINY_POINTS = [(0.0, 1.0), (1.0, 0.0), (-1.0, 0.5), (0.5, -0.5)]
TINY_LABELS = [1, -1, 1, -1]


@pytest.mark.parametrize("y, s, expected", [(1, 5.0, 0.0), (1, -1.0, 2.0)])
def test_hinge(y: int, s: float, expected: float) -> None:
    assert hinge(Value(s), y).data == pytest.approx(expected)


def test_one_step_lowers_loss() -> None:
    random.seed(0)
    model = MLP(2, [4, 1])
    before = loss_fn(model, TINY_POINTS, TINY_LABELS).data
    train_step(model, TINY_POINTS, TINY_LABELS, lr=0.1)
    after = loss_fn(model, TINY_POINTS, TINY_LABELS).data
    assert after < before


def test_grads_are_zero_after_step() -> None:
    random.seed(0)
    model = MLP(2, [4, 1])
    train_step(model, TINY_POINTS, TINY_LABELS, lr=0.1)
    assert all(p.grad == 0 for p in model.parameters())


def test_full_training_reaches_95_percent() -> None:
    model, _, points, labels = train(log_every=0)
    assert accuracy(model, points, labels) >= 0.95
