# autograd-engine

A from-scratch reverse-mode autodiff engine, built to understand backprop:
first on scalars, then on NumPy tensors, ending with an MLP trained on MNIST.

Specs are written by hand in `specs/`; code is generated from them and reviewed
line by line. Hand-written drills live in `drills/`. See [PLAN.md](PLAN.md).

## Layout

```
specs/      piece-by-piece specs (math + tests)
engine.py   scalar Value
tensor.py   NumPy Tensor
nn.py       Neuron, Layer, MLP
tests/      checked against PyTorch and finite differences
drills/     written by hand, no AI
notes.md    predictions, results, surprises
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```
