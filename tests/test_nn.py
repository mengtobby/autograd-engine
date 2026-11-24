"""Tests from specs/nn.md "Tests I want"."""

import random

import pytest

from engine import Value
from gradcheck import gradcheck
from nn import MLP


def test_param_count_is_17() -> None:
    params = MLP(2, [4, 1]).parameters()
    assert len(params) == 17
    assert all(isinstance(p, Value) for p in params)


def test_init_range_and_not_all_equal() -> None:
    random.seed(0)
    values = [p.data for p in MLP(2, [16, 16, 1]).parameters()]
    assert all(-1 <= v <= 1 for v in values)
    assert len(set(values)) > 1


def test_every_parameter_gets_a_gradient() -> None:
    random.seed(1)
    template = MLP(2, [3, 1])
    start = [p.data for p in template.parameters()]

    def f(weights: list[Value]) -> Value:
        model = MLP(2, [3, 1])
        neurons = [n for layer in model.layers for n in layer.neurons]
        it = iter(weights)
        for n in neurons:  # swap in gradcheck's Values, in parameters() order
            n.w = [next(it) for _ in n.w]
            n.b = next(it)
        out = model([0.5, -1.2])
        return out * out

    gradcheck(f, start)


def test_wrong_input_length_raises() -> None:
    with pytest.raises(ValueError):
        MLP(2, [4, 1])([1.0, 2.0, 3.0])


def test_output_layer_is_linear() -> None:
    model = MLP(2, [1])  # a single (output) layer
    neuron = model.layers[0].neurons[0]
    x = [0.3, -0.8]
    expected = sum(w.data * xi for w, xi in zip(neuron.w, x)) + neuron.b.data
    assert model(x).data == pytest.approx(expected)


def test_relu_option_and_unknown_activation() -> None:
    model = MLP(2, [5, 1], activation="relu")
    hidden = model.layers[0]([0.3, -0.8])
    assert all(h._op == "relu" for h in hidden)
    with pytest.raises(ValueError):
        MLP(2, [5, 1], activation="sigmoid")
