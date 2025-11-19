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

train_losses = []
train_ious = []
test_epochs = []
test_ious = []
test_losses = []
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

# ---------- helpers ----------

def read_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        ids = [l.strip() for l in f if l.strip()]
    return ids

tfms = A.Compose([
    A.Resize(1024, 1024),
    ToTensorV2()
])

class OverfitDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        img_path = os.path.join(IMG_DIR, sid + ".png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for {sid}: {img_path}")

        mask_path = os.path.join(MASK_DIR, sid + "_mask.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {sid}: {mask_path}")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path).astype(np.uint8)

        aug = tfms(image=img, mask=mask)
        # TODO this does normalization between 0-1, do we need -1, 1?
        # SOL: No normalization for now
        x = aug["image"].float() #/ 255.0          # [3,H,W]
                                                            #TODO do we need normalization in first phase?
        y = aug["mask"].unsqueeze(0).float()      # [1,H,W]
        return x, y

def dice_loss(logits, targets, eps=1e-6):  #TODO why use this loss? why not use iou_score as loss?
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

def main():
    # overfit subset
    overfit_ids = read_ids("ids/val_ids_subsample.txt")
    print("Loaded", len(overfit_ids), "overfit samples")
    if len(overfit_ids) == 0:
        raise RuntimeError("val_ids_subsample.txt is empty. Check how you generated it.")

    # test split
    test_ids = read_ids("ids/test_ids_subsample.txt")
    print("Loaded", len(test_ids), "test samples")
    if len(test_ids) == 0:
        raise RuntimeError("test_ids_subsample.txt is empty.")

    train_ds = OverfitDataset(overfit_ids)
    test_ds = OverfitDataset(test_ids)

    batch_size = min(4, len(train_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True #TODO what is pin memory?
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",  #TODO motivation why we chose this as starter
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # TODO motivate why using this optimizer with this learning rate
    # Sol: move from Adam to basic SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # TODO what is this
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    EPOCHS = 100
    # best_train_iou = 0.0
    # best_test_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ----- train on overfit subset -----
        model.train()
        total_loss = 0.0
        n = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            x = x.to(device, non_blocking=True) #TODO what does non blocking do
            y = y.to(device, non_blocking=True)

            # TODO what does set to none do
            # Sol: some pytorch optim, move to end
            # optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = combined_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        avg_loss = total_loss / n

        # ----- compute train IoU (still using current model in train mode) -----
        total_iou_train = 0.0
        n_train = 0
        #TODO if we choose iou as loss function no need to do this again
        #Sol iou is evaluation metric, but we still do not need this at every epoch
        # with torch.no_grad():
        #     for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        #         x = x.to(device)
        #         y = y.to(device)
        #         logits = model(x)
        #         total_iou_train += iou_score(logits, y) * x.size(0)
        #         n_train += x.size(0)
        # train_iou = total_iou_train / n_train if n_train > 0 else 0.0

        train_losses.append(avg_loss)
        #train_ious.append(train_iou)


        # log like the example: both train + test every epoch
        if epoch == 1 or epoch % 10 == 0:
            # ----- compute test IoU with model.eval() -----
            train_iou, _ = evaluate_iou(model, train_loader, device)
            test_iou, test_loss = evaluate_iou(model, test_loader, device)

            #best_train_iou = max(best_train_iou, train_iou)
            #best_test_iou = max(best_test_iou, test_iou)
            train_ious.append(train_iou)
            test_ious.append(test_iou)
            test_epochs.append(epoch)
            test_losses.append(test_loss)
            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                f"train_IoU={train_iou:.4f} | test_IoU={test_iou:.4f} | "
                f"train_loss={avg_loss:.4f} | test_loss={test_loss:.4f}"
            )

    torch.save(model.state_dict(), "unet_overfit_subset_on_subsample_test_valid.pt")
    print("Saved unet_overfit_subset_on_subsample_test_valid.pt")
    #print(f"Best train IoU: {best_train_iou:.4f} | Best test IoU: {best_test_iou:.4f}")
    print("If train IoU >> test IoU, you've clearly demonstrated overfitting.")
    plot_loss(train_losses, val_losses=None,
              save_path="plots/train_loss_curve_on_subsample_test_valid_2.png")

    plot_loss(test_losses, val_losses=None,
              save_path="plots/test_loss_curve_on_subsample_test_valid_2.png")

    # plot_iou(train_ious, test_ious, test_epochs,
    #          save_path="plots/iou_curve_on_subsample_test_valid.png")
    plot_iou(train_ious, test_ious,
             save_path="plots/iou_curve_on_subsample_test_valid_2.png")


if __name__ == "__main__":
    main()
