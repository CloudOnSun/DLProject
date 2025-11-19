import os
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt

Number = Union[int, float]

def _to_list_clean(x: Optional[Iterable]) -> list:
    """Convert tensors/arrays/iterables to a Python list and drop NaN/Inf."""
    if x is None:
        return []
    try:
        # torch/tf/np support: rely on list() fallback
        x = list(x)
    except TypeError:
        x = [x]
    clean = []
    for v in x:
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue
        except Exception:
            pass
        clean.append(float(v))
    return clean


def _finalize_plot(
    title: str = "",
    xlabel: str = "Epoch",
    ylabel: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✅ Plot saved to: {os.path.abspath(save_path)}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(
    lrs: Sequence[Number],
    steps: Optional[Sequence[Number]] = None,
    title: str = "Learning Rate Schedule",
    save_path: Optional[str] = None,
    show: bool = True,
):
    lrs = _to_list_clean(lrs)
    steps = _to_list_clean(steps) if steps is not None else list(range(len(lrs)))
    # length guard
    n = min(len(steps), len(lrs))
    if len(steps) != len(lrs):
        print(f"⚠️ plot_learning_rate: length mismatch; truncating to {n}.")
    plt.figure(figsize=(7, 5))
    plt.plot(steps[:n], lrs[:n], label="learning rate", color="C2")
    _finalize_plot(title=title, xlabel="Step", ylabel="Learning rate", save_path=save_path, show=show)


def plot_loss(
    train_losses: Sequence[Number],
    val_losses: Optional[Sequence[Number]] = None,
    title: str = "Loss over epochs",
    save_path: Optional[str] = None,
    show: bool = True,
):
    train_losses = _to_list_clean(train_losses)
    plt.figure(figsize=(7, 5))
    if train_losses:
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, label="train loss", color="C0")

    if val_losses is not None:
        val_losses = _to_list_clean(val_losses)
        if val_losses:
            val_epochs = list(range(1, len(val_losses) + 1))
            n = min(len(val_epochs), len(val_losses))
            plt.plot(val_epochs[:n], val_losses[:n], label="val loss", color="C1")

    _finalize_plot(title=title, xlabel="Epoch", ylabel="Loss", save_path=save_path, show=show)


def plot_iou(
    train_ious: Sequence[Number],
    test_ious: Optional[Sequence[Number]] = None,
    test_epochs: Optional[Sequence[Number]] = None,
    *,
    test_every: Optional[int] = None,
    title: str = "IoU over epochs",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot IoU curves for training and test sets.

    Args:
        train_ious: IoU logged every epoch (length = #epochs).
        test_ious: IoU logged less frequently (e.g., every N epochs).
        test_epochs: Explicit epoch indices for each test IoU (e.g., [10,20,30,...]).
        test_every: If you logged every N epochs and didn't store test_epochs,
                    set this (e.g., test_every=10) and epochs will be [N,2N,3N,...].
    """
    train_ious = _to_list_clean(train_ious)
    test_ious = _to_list_clean(test_ious)

    plt.figure(figsize=(7, 5))
    # train
    if train_ious:
        epochs = list(range(1, len(train_ious) + 1))
        plt.plot(epochs, train_ious, label="train IoU", color="C0")

    # test
    if test_ious:
        if test_epochs is not None:
            # use provided epochs, but guard lengths
            te = _to_list_clean(test_epochs)
            m = min(len(te), len(test_ious))
            if len(te) != len(test_ious):
                print(f"⚠️ plot_iou: test_epochs ({len(te)}) != test_ious ({len(test_ious)}); truncating to {m}.")
            x = te[:m]
            y = test_ious[:m]
        elif test_every is not None:
            # infer epochs from periodic logging
            # start at test_every, then 2*test_every, ...
            x = [k * test_every for k in range(1, len(test_ious) + 1)]
            y = test_ious
        else:
            # assume contiguous if nothing provided
            x = list(range(1, len(test_ious) + 1))
            y = test_ious

        plt.plot(x, y, label="test IoU", color="C1")

    _finalize_plot(title=title, xlabel="Epoch", ylabel="IoU", save_path=save_path, show=show)
