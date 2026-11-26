"""Train a Tensor MLP on MNIST. Implements specs/train_mnist.md.

Run: python train_mnist.py"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from loss import cross_entropy
from tensor import Tensor

Layer = tuple[Tensor, Tensor]  # (W (n_in, n_out), b (n_out,))
Data = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

N_CLASSES = 10
N_TRAIN = 60_000
CACHE = "mnist.npz"


# ----- data -----

def one_hot(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    return np.eye(n_classes)[labels]


def load_mnist(data_home: str = "data") -> Data:
    """(X_train, y_train, X_test, y_test). Pixels ÷ 255; labels are ints 0–9."""
    cache = Path(data_home) / CACHE
    if not cache.exists():
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(
            "mnist_784", version=1, as_frame=False, data_home=data_home,
            return_X_y=True, parser="liac-arff",
        )
        np.savez_compressed(cache, X=X.astype(np.uint8), y=y.astype(np.int64))
    saved = np.load(cache)
    X, y = saved["X"] / 255.0, saved["y"]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


# ----- model -----

def init_params(sizes: Sequence[int], rng: np.random.Generator) -> list[Layer]:
    """W = uniform(-1, 1) / √n_in, b = 0, for each pair of sizes."""
    return [
        (Tensor(rng.uniform(-1, 1, size=(n_in, n_out)) / np.sqrt(n_in)), Tensor(np.zeros(n_out)))
        for n_in, n_out in zip(sizes[:-1], sizes[1:])
    ]


def parameters(layers: Sequence[Layer]) -> list[Tensor]:
    return [p for W, b in layers for p in (W, b)]


def forward(layers: Sequence[Layer], X: np.ndarray) -> Tensor:
    """X @ W + b per layer; tanh on hidden layers, the last layer linear (logits)."""
    h = Tensor(X)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = h.tanh()
    return h


def accuracy(layers: Sequence[Layer], X: np.ndarray, y: np.ndarray, chunk: int = 1000) -> float:
    """Forward only, in chunks. Prediction = argmax of the logits."""
    correct = sum(
        int((forward(layers, X[i:i + chunk]).data.argmax(axis=1) == y[i:i + chunk]).sum())
        for i in range(0, len(X), chunk)
    )
    return correct / len(X)


# ----- training -----

def train_step(layers: Sequence[Layer], X: np.ndarray, Y: np.ndarray, lr: float) -> float:
    """Forward + loss → backward → update → zero grads. Returns the batch loss."""
    loss = cross_entropy(forward(layers, X), Y)
    loss.backward()
    for p in parameters(layers):
        p.data -= lr * p.grad
        p.grad = np.zeros_like(p.data)
    return loss.data.item()


def train(
    epochs: int = 10,
    lr: float = 0.1,
    batch_size: int = 64,
    hidden: int = 128,
    seed: int = 0,
    n_train: int | None = None,
    log_every: int = 100,
    data: Data | None = None,
) -> tuple[list[Layer], dict[str, list[float]]]:
    X_train, y_train, X_test, y_test = data if data is not None else load_mnist()
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    rng = np.random.default_rng(seed)
    layers = init_params((X_train.shape[1], hidden, N_CLASSES), rng)
    history: dict[str, list[float]] = {"loss": [], "test_acc": [], "epoch_seconds": []}

    start = cross_entropy(forward(layers, X_train[:batch_size]), one_hot(y_train[:batch_size]))
    if log_every:
        print(f"starting loss {start.data.item():.4f} (ln 10 = {np.log(10):.4f})")

    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(len(X_train))  # shuffle every epoch
        running: list[float] = []
        for i in range(0, len(order), batch_size):  # last batch may be smaller
            idx = order[i:i + batch_size]
            loss = train_step(layers, X_train[idx], one_hot(y_train[idx]), lr)
            history["loss"].append(loss)
            running.append(loss)
            step += 1
            if log_every and step % log_every == 0:
                print(f"step {step:5d}  loss {np.mean(running):.4f}")
                running = []
        seconds = time.perf_counter() - t0
        acc = accuracy(layers, X_test, y_test)
        history["test_acc"].append(acc)
        history["epoch_seconds"].append(seconds)
        if log_every:
            print(f"epoch {epoch:2d}  test accuracy {acc:.2%}  ({seconds:.1f} s)")
    return layers, history


def save_plot(history: dict[str, list[float]], path: str = "mnist_loss.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], linewidth=0.5)
    ax.set(xlabel="step", ylabel="training loss (batch)", title="MNIST training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    _, hist = train()
    save_plot(hist)
