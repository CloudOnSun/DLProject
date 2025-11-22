import os

import torch
from matplotlib import pyplot as plt
from torch import nn

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
        if isinstance(module, nn.ReLU):
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

