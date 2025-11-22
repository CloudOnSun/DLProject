import os
import math
from typing import Iterable, Optional, Sequence, Tuple, Union
import cv2

import torch
import numpy as np

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


def plot_loss(train_losses, val_losses, epochs, save_path=None):

    train_losses = _to_list_clean(train_losses)
    plt.figure(figsize=(8,5))

    plt.plot(epochs, train_losses, label="Train Loss", marker="o")

    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Test Loss", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_iou(train_ious, test_ious, epochs, save_path=None):
    plt.figure(figsize=(8,5))

    plt.plot(epochs, train_ious, label="Train IoU", marker="o")
    plt.plot(epochs, test_ious, label="Test IoU", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU Curve")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

def predict_and_show(model: torch.nn.Module, tile_id: str, tfms,  device="cuda",
                     threshold: float = 0.5, save_path: str | None = None):
    """
    Given a model and an image id (tile_id):
    - loads the image
    - runs the model to predict a building mask
    - shows the image with the predicted mask overlaid in red.

    Args:
        model: a UNet-like model that maps (B,3,H,W) -> (B,1,H,W) logits or probs
        tile_id: e.g. "guatemala-volcano_00000006_pre_disaster"
        device: cuda or cpu
        threshold: probability threshold for binary mask
        save_path: optional path to save the overlay image (PNG)
    """
    model.eval()
    model.to(device)

    # ---- Load & preprocess image ----
    img_path = os.path.join("data/images", tile_id + ".png")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


    # ---- Forward pass ----
    with torch.no_grad():
        aug = tfms(image=img)
        x = aug["image"].float().to(device)
        pred = model(x.unsqueeze(0))
        # If model returns logits, apply sigmoid
        if pred.shape[1] == 1:
            prob = torch.sigmoid(pred)
            prob = prob.squeeze(0).squeeze(0) # (H,W)
        else:
            # if C>1, assume channel 1 is building; adapt if needed
            prob = torch.softmax(pred, dim=1)[:, 1, ...].squeeze(0)

    # Optionally resize back if model changes resolution
    # (here we assume output is same size as input)

    prob_np = prob.cpu().numpy()              # (H,W)
    mask_bin = (prob_np >= threshold).astype(np.uint8)

    # ---- Prepare overlay ----
    h, w = prob_np.shape

    rgba_mask = np.zeros((h, w, 4), dtype=np.float32)
    rgba_mask[mask_bin == 1] = [1.0, 0.0, 0.0, 0.6]   # red with alpha=0.6
    imgplt = plt.imread(img_path)

    # ---- Plot ----
    plt.figure(figsize=(8, 8))
    plt.imshow(imgplt)
    plt.imshow(rgba_mask)
    plt.axis("off")
    plt.title(f"Predicted mask overlay: {tile_id}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved overlay to {save_path}")

    plt.show()
