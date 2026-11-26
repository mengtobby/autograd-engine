"""Tests from specs/train_mnist.md "Tests I want".

Needs MNIST cached in data/ (the first load downloads it)."""

import math

import numpy as np
import pytest

from gradcheck import tensor_gradcheck
from loss import cross_entropy
from train_mnist import (
    accuracy,
    forward,
    init_params,
    load_mnist,
    one_hot,
    parameters,
    train,
    train_step,
)


@pytest.fixture(scope="module")
def mnist():
    return load_mnist()


def test_parameter_count() -> None:
    layers = init_params((784, 128, 10), np.random.default_rng(0))
    assert sum(p.data.size for p in parameters(layers)) == 101_770


def test_starting_loss_is_ln10(mnist) -> None:
    X_train, y_train, _, _ = mnist
    layers = init_params((784, 128, 10), np.random.default_rng(0))
    loss = cross_entropy(forward(layers, X_train[:64]), one_hot(y_train[:64]))
    assert abs(loss.data.item() - math.log(10)) < 0.1


def test_gradcheck_model_and_loss() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(2, 784))  # 2 images
    Y = one_hot(np.array([3, 8]))
    layers = init_params((784, 4, 10), rng)  # hidden layer of 4
    arrays = [p.data for p in parameters(layers)]

    def f(ts):
        return cross_entropy(forward([(ts[0], ts[1]), (ts[2], ts[3])], X), Y)

    tensor_gradcheck(f, arrays)


def test_one_step_lowers_loss(mnist) -> None:
    X_train, y_train, _, _ = mnist
    X, Y = X_train[:64], one_hot(y_train[:64])
    layers = init_params((784, 128, 10), np.random.default_rng(0))
    before = cross_entropy(forward(layers, X), Y).data.item()
    train_step(layers, X, Y, lr=0.1)
    after = cross_entropy(forward(layers, X), Y).data.item()
    assert after < before


def test_short_run_beats_80_percent(mnist) -> None:
    X_train, y_train, X_test, y_test = mnist
    layers, _ = train(epochs=1, n_train=5000, log_every=0, data=mnist)
    assert accuracy(layers, X_test, y_test) > 0.80
