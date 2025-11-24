import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from overfit_method import iou_score


def plot_hist(hs,xrange=(-1,1),avg=None,sd=None):
  plt.figure(figsize=(20,3))
  for layer in range(len(hs)):
    plt.subplot(1,len(hs),layer+1)
    activations = hs[layer].detach().cpu().numpy().flatten()
    plt.hist(activations, bins=20, range=xrange)

    title = 'Layer ' + str(layer+1)
    if avg:
      title += '\n' + "mean {0:.2f}".format(avg[layer])
    if sd:
      title += '\n' + "std {0:.4f}".format(sd[layer])

    plt.title(title)

def plot_hist_chunked(hs, names, chunk_size=6, avg=None, sd=None):
    n_layers = len(hs)
    for start in range(0, n_layers, chunk_size):
        end = min(start + chunk_size, n_layers)
        plt.figure(figsize=(22, 4))

        for idx, layer in enumerate(range(start, end), 1):
            plt.subplot(1, end-start, idx)
            activations = hs[layer].detach().cpu().numpy().flatten()
            plt.hist(activations, bins=20)

            title = names[layer]
            if avg:
                title += f"\nmean={avg[layer]:.2f}"
            if sd:
                title += f"\nstd={sd[layer]:.2f}"

            plt.title(title, fontsize=8)

        plt.tight_layout()
        plt.show()

def plot_hist_sampled(hs, names, step=7, avg=None, sd=None, save_path=None):
    # Select indices: 0, step, 2*step, ...
    sampled_indices = list(range(0, len(hs), step))

    n = len(sampled_indices)
    plt.figure(figsize=(3 * n, 3))

    for i, layer_idx in enumerate(sampled_indices):
        plt.subplot(1, n, i + 1)
        activ = hs[layer_idx].detach().cpu().numpy().flatten()
        plt.hist(activ, bins=20)

        title = names[layer_idx]
        if avg:
            title += f"\nmean={avg[layer_idx]:.5f}"
        if sd:
            title += f"\nstd={sd[layer_idx]:.5f}"

        plt.title(title, fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()




def show_stats(stats, model_name):

    print('loss', stats['loss'].item())
    print('accuracy', stats['accuracy'], '\n')

    # -----------------------
    # GRADIENT HISTOGRAMS
    # -----------------------
    save_path = f"stats/{model_name}_gradients.png"
    plot_hist_sampled(
        stats["grads"],
        stats["names"],
        step=7,
        avg=stats["gradient_mean"],
        sd=stats["gradient_std"],
        save_path=save_path
    )

    # -----------------------
    # ACTIVATION HISTOGRAMS
    # -----------------------

    save_path = f"stats/{model_name}_activations.png"
    plot_hist_sampled(
        stats["activations"],
        stats["activation_names"],
        step=4,
        avg=stats["activation_mean"],
        sd=stats["activation_std"],
        save_path=save_path
    )
    plt.close()

def get_layer_stats(x,absolute=False):
  avg = []
  std = []
  for layer in range(len(x)):
    if absolute:
      avg.append(x[layer].abs().mean().detach().cpu().numpy())
    else:
      avg.append(x[layer].mean().detach().cpu().numpy())

    std.append(x[layer].std().detach().cpu().numpy())

  return avg, std


def get_stats(model, dataloader, loss_func, device="cuda"):
    model.train()  # we want gradients

    # ---- grab one batch ----
    dataiter = iter(dataloader)
    images, masks = next(dataiter)
    images = images.to(device)
    masks = masks.to(device)

    # ---- containers for activations / layer metadata ----
    activations = {}
    layer_names = []
    conv_modules = []     # to later fetch their weight gradients
    hooks = []

    # ---- register forward hooks on Conv2d layers ----
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_names.append(name)
            conv_modules.append(module)
        if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):
            def hook_fn(mod, inp, out, layer=name):
                # store only first call per module
                if layer not in activations:
                    activations[layer] = out

            hooks.append(module.register_forward_hook(hook_fn))

    # ---- forward pass ----
    model.zero_grad()
    scores = model(images)              # predictions (UNet output)
    loss = loss_func(scores, masks)
    acc = iou_score(scores, masks)

    # ---- backward pass (grads) ----
    loss.backward()

    # ---- collect gradients for the same layers (their weights) ----
    gradients = []
    for m in conv_modules:
        if m.weight.grad is not None:
            gradients.append(m.weight.grad)
        else:
            # Just in case something has no grad
            gradients.append(torch.zeros_like(m.weight))

    # ---- remove hooks to avoid side-effects later ----
    for h in hooks:
        h.remove()

    # ---- compute stats ----
    activation_mean, activation_std = get_layer_stats(list(activations.values()))
    gradient_mean, gradient_std = get_layer_stats(gradients, absolute=True)

    stats = {
        'loss': loss,
        'accuracy': acc,
        'names': layer_names,
        'activation_names': list(activations.keys()),
        'grads': gradients,
        'activations': list(activations.values()),
        'activation_mean': activation_mean,
        'activation_std': activation_std,
        'gradient_mean': gradient_mean,
        'gradient_std': gradient_std
    }

    return stats


def save_error_maps_per_image(
    model,
    loader,
    device,
    save_dir="plots/error_maps_per_image",
    threshold=0.5,
    denorm_mode="imagenet"
):
    """
    For each image in the loader:
      - Compute TP / FP / FN / TN masks
      - Save a visualization:
           Left: original RGB image
           Right: error overlay (TP green, FP red, FN blue)
      - Also return per-image confusion stats

    Args:
        model: trained segmentation model
        loader: DataLoader for test set
        device: cuda or cpu
        save_dir: directory where per-image PNGs will be saved
        threshold: threshold for binary prediction
        denorm_mode: use "imagenet" to reverse normalization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    def denormalize(x):
        x = x.numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        x = x * std + mean
        x = np.clip(x, 0, 1)
        return np.transpose(x, (1,2,0))  # CHW -> HWC

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_results = []  # list of dicts, per-image confusion and IoU

    with torch.no_grad():
        idx_global = 0
        for x, y in tqdm(loader, desc="Per-image error maps"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Loop over batch (usually batch size=1 here)
            B = x.size(0)
            for b in range(B):
                img = x[b].cpu()
                gt = (y[b,0] > 0.5).cpu().numpy()
                pr = (preds[b,0] > 0.5).cpu().numpy()

                # confusion maps
                tp = (pr == 1) & (gt == 1)
                fp = (pr == 1) & (gt == 0)
                fn = (pr == 0) & (gt == 1)
                tn = (pr == 0) & (gt == 0)

                # compute per-image confusion counts
                TP = tp.sum()
                FP = fp.sum()
                FN = fn.sum()
                TN = tn.sum()
                IoU = TP / (TP + FP + FN + 1e-6)

                all_results.append({
                    "index": idx_global,
                    "TP": int(TP),
                    "FP": int(FP),
                    "FN": int(FN),
                    "TN": int(TN),
                    "IoU": float(IoU),
                })

                # visualization mask
                H, W = gt.shape
                overlay = np.zeros((H, W, 3), dtype=np.float32)
                overlay[tp] = [0.00, 1.00, 0.00]  # green
                overlay[fp] = [1.00, 0.00, 0.00]  # red
                overlay[fn] = [0.00, 0.00, 1.00]  # blue

                # original image (denormalized)
                img_rgb = denormalize(img)

                # plot
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(img_rgb)
                axs[0].set_title("Image")
                axs[0].axis("off")

                axs[1].imshow(img_rgb)
                axs[1].imshow(overlay, alpha=0.5)
                axs[1].set_title(
                    f"TP/FP/FN overlay\nIoU={IoU:.3f}, TP={TP}, FP={FP}, FN={FN}"
                )
                axs[1].axis("off")

                out_path = os.path.join(save_dir, f"error_map_{idx_global:04d}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)

                idx_global += 1

    return all_results
