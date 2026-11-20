# autograd-engine

A from-scratch reverse-mode autodiff engine, built to understand backprop:
first on scalars, then on NumPy tensors, ending with an MLP trained on MNIST.

Built AI-natively: I do the derivations, design decisions, predictions and
review; an AI writes every file from what I specify. Specs live in `specs/`,
recall-drill builds in `drills/`. See [PLAN.md](PLAN.md).

## Layout

```
specs/      piece-by-piece specs (math + tests), dictated then approved
engine.py   scalar Value
tensor.py   NumPy Tensor
nn.py       Neuron, Layer, MLP
tests/      checked against PyTorch and finite differences
drills/     literal builds of from-memory specs (recall drills)
notes.md    predictions, results, surprises
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```
