import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from plot_utils import plot_loss, plot_iou
import torch.nn as nn
import torch.nn.init as init


# ---------- Data Loading ----------

def norm_collate(batch):
    # batch is a list of (x, y) from your Dataset
    xs, ys = zip(*batch)            # tuples of tensors
    xs = torch.stack(xs, dim=0)     # [B, 3, H, W]
    ys = torch.stack(ys, dim=0)     # [B, 1, H, W]

    # your current x is in [0, 255] as float
    xs = xs / 255.0
    xs = xs * 2.0 - 1.0             # now in [-1, 1]

    return xs, ys

def read_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        ids = [l.strip() for l in f if l.strip()]
    return ids

tfms_basic = A.Compose([
    A.Resize(1024, 1024),
    ToTensorV2()
])

tfms_random_crop = A.Compose([
    A.RandomCrop(256, 256),
    ToTensorV2()
])

tfms_random_crop_normalized = A.Compose([
    A.RandomCrop(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0),
    ToTensorV2()
])

tfms_normalized = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0),
    ToTensorV2()
])

tfms_random_crop_normalized_encoder_vals = A.Compose([
    A.RandomCrop(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,  # assumes uint8 images in [0, 255]
    ),
    ToTensorV2()
])

tfms_val_normalized_encoder_vals = A.Compose([
    A.CenterCrop(256, 256),  # or A.Resize(...) / nothing, depending on your setup
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])



def init_kaiming_for_conv(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # He/Kaiming init for ReLU
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        # Standard BN init: gamma=1, beta=0
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        # For any linear layers in the head, Xavier is a decent default
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class OverfitDataset(Dataset):
    def __init__(self, ids, tfms):
        self.ids = ids
        self.tfms = tfms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        IMG_DIR = "data/images"
        MASK_DIR = "data/masks"
        sid = self.ids[idx]

        img_path = os.path.join(IMG_DIR, sid + ".png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for {sid}: {img_path}")

        mask_path = os.path.join(MASK_DIR, sid + "_mask.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {sid}: {mask_path}")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path).astype(np.uint8)

        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        x = img.float()  # if not normalized yet
        y = mask.unsqueeze(0).float()      # [1,H,W]
        return x, y


# -------------------- LOSS -----------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()

def combined_loss(logits, targets):
    bce = nn.BCEWithLogitsLoss()
    return bce(logits, targets) + dice_loss(logits, targets)

def iou_score(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    t = (targets > 0.5).float()
    inter = (preds * t).sum(dim=(1,2,3))
    union = ((preds + t) > 0).float().sum(dim=(1,2,3))
    return ((inter + eps) / (union + eps)).mean().item()

def evaluate_iou(model, loader, device):
    """
    Compute mean IoU on a given loader in eval mode.
    Restores previous training/eval state afterwards.
    """
    was_training = model.training
    model.eval()

    total_iou = 0.0
    n = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = combined_loss(logits, y)
            total_iou += iou_score(logits, y) * x.size(0)
            n += x.size(0)

    if was_training:
        model.train()

    return total_iou / n if n > 0 else 0.0, loss

# ---------- main ----------

def main(train_loader, test_loader, model, optimizer, epochs, loss_func,
         model_file, train_loss_file, test_loss_file, iou_file, device, scheduler=None, loss_finder=None):
    train_losses = []
    train_ious = []
    test_epochs = []
    test_ious = []
    test_losses = []
    train_history = []

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        # ----- train on overfit subset -----
        model.train()
        total_loss = 0.0
        n = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_func(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            if loss_finder:
                train_history.append(loss.item())

        avg_loss = total_loss / n

        if epoch == 1 or epoch % 5 == 0:
            # ----- compute test IoU with model.eval() -----
            train_iou, _ = evaluate_iou(model, train_loader, device)
            test_iou, test_loss = evaluate_iou(model, test_loader, device)

            train_ious.append(train_iou)
            test_ious.append(test_iou)
            test_epochs.append(epoch)
            test_losses.append(test_loss)
            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                f"train_IoU={train_iou:.4f} | test_IoU={test_iou:.4f} | "
                f"train_loss={avg_loss:.4f} | test_loss={test_loss:.4f}"
            )

        if scheduler:
            scheduler.step()

        train_losses.append(avg_loss)

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save(model.state_dict(), model_file)
    print("Saved " + model_file)
    plot_loss(train_losses, val_losses=None, epochs=list(range(1, len(train_losses)+1)), save_path=train_loss_file)

    plot_loss(test_losses, val_losses=None, epochs=test_epochs, save_path=test_loss_file)

    plot_iou(train_ious, test_ious=test_ious, epochs=test_epochs, save_path=iou_file)

    if loss_finder:
        return train_history

    return None