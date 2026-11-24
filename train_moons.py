"""Train an MLP on the two moons. Implements specs/train_moons.md.

Run: python train_moons.py"""

from __future__ import annotations

import random
from collections.abc import Sequence

from engine import Value
from nn import MLP

Point = tuple[float, float]


def load_data(n: int = 100, noise: float = 0.1, seed: int = 0) -> tuple[list[Point], list[int]]:
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    points = [(float(a), float(b)) for a, b in X]
    labels = [2 * int(label) - 1 for label in y]  # 0/1 -> -1/+1
    return points, labels


def hinge(score: Value, y: int) -> Value:
    """max(0, 1 - y*s): 0 once on the right side by at least 1."""
    return (1 - y * score).relu()


def loss_fn(model: MLP, points: Sequence[Point], labels: Sequence[int]) -> Value:
    """Mean hinge loss over all points."""
    losses = [hinge(model(p), y) for p, y in zip(points, labels)]
    return sum(losses) * (1.0 / len(losses))


def accuracy(model: MLP, points: Sequence[Point], labels: Sequence[int]) -> float:
    correct = sum((model(p).data > 0) == (y > 0) for p, y in zip(points, labels))
    return correct / len(labels)


def train_step(model: MLP, points: Sequence[Point], labels: Sequence[int], lr: float) -> float:
    loss = loss_fn(model, points, labels)  # 1. forward + loss
    loss.backward()  # 2. backward
    for p in model.parameters():
        p.data -= lr * p.grad  # 3. update
    for p in model.parameters():
        p.grad = 0  # 4. zero grads
    return loss.data


def train(
    steps: int = 100,
    lr: float = 0.1,
    sizes: Sequence[int] = (16, 16, 1),
    activation: str = "tanh",
    seed: int = 0,
    log_every: int = 10,
) -> tuple[MLP, list[float], list[Point], list[int]]:
    random.seed(seed)
    points, labels = load_data()
    model = MLP(2, sizes, activation)
    history = []
    for step in range(steps):
        history.append(train_step(model, points, labels, lr))
        if log_every and (step % log_every == 0 or step == steps - 1):
            acc = accuracy(model, points, labels)
            print(f"step {step:3d}  loss {history[-1]:.4f}  accuracy {acc:.0%}")
    return model, history, points, labels


def save_plots(model: MLP, history: list[float], points: list[Point], labels: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs, ys = np.meshgrid(np.linspace(-1.5, 2.5, 50), np.linspace(-1.0, 1.5, 50))
    scores = np.array([[model((float(x), float(y))).data for x, y in zip(rx, ry)] for rx, ry in zip(xs, ys)])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.contourf(xs, ys, scores > 0, alpha=0.3, cmap="coolwarm")
    px, py = zip(*points)
    ax.scatter(px, py, c=labels, cmap="coolwarm", edgecolors="k", s=25)
    ax.set_title("Decision boundary (score > 0 vs < 0)")
    fig.savefig("moons_boundary.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history)
    ax.set_xlabel("step")
    ax.set_ylabel("mean hinge loss")
    ax.set_title("Training loss")
    fig.savefig("moons_loss.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    model, history, points, labels = train()
    save_plots(model, history, points, labels)
    print("saved moons_boundary.png and moons_loss.png")
