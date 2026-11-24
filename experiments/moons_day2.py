"""Day 2 Block 6 experiments: one change from the baseline per run.

Run from the repo root: python -m experiments.moons_day2"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor

from nn import MLP
from train_moons import accuracy, load_data, loss_fn

BASELINE = {"lr": 0.1, "sizes": (16, 16, 1), "activation": "tanh", "zero_grads": True}

RUNS = {
    "baseline": {},
    "E1 lr 0.001": {"lr": 0.001},
    "E1 lr 1.0": {"lr": 1.0},
    "E1 lr 10": {"lr": 10.0},
    "E2 no hidden [1]": {"sizes": (1,)},
    "E2 one hidden [16,1]": {"sizes": (16, 1)},
    "E3 relu": {"activation": "relu"},
    "E4 no zero grads": {"zero_grads": False},
}
CHECKPOINTS = (0, 10, 50, 99)


def run(name: str) -> tuple[str, list[float], float]:
    cfg = {**BASELINE, **RUNS[name]}
    random.seed(0)
    points, labels = load_data()
    model = MLP(2, cfg["sizes"], cfg["activation"])
    losses = []
    for _ in range(100):
        loss = loss_fn(model, points, labels)
        loss.backward()
        for p in model.parameters():
            p.data -= cfg["lr"] * p.grad
        if cfg["zero_grads"]:
            for p in model.parameters():
                p.grad = 0
        losses.append(loss.data)
        if not math.isfinite(loss.data):
            break
    acc = accuracy(model, points, labels) if math.isfinite(losses[-1]) else float("nan")
    return name, losses, acc


def fmt(x: float) -> str:
    return f"{x:9.3g}" if math.isfinite(x) else "      nan"


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run, RUNS))
    print(f"{'run':22} | " + " | ".join(f"loss@{c:<4}" for c in CHECKPOINTS) + " | accuracy")
    print("-" * 82)
    for name, losses, acc in results:
        cells = [fmt(losses[c]) if c < len(losses) else "  stopped" for c in CHECKPOINTS]
        acc_s = f"{acc:.0%}" if math.isfinite(acc) else "nan"
        print(f"{name:22} | " + " | ".join(cells) + f" | {acc_s}")
