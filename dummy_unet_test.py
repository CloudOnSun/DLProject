# dummy_unet_test.py
import os, random, cv2, numpy as np, torch
import torch.nn as nn
from tqdm import tqdm
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Paths ----
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
TEST_IDS = [l.strip() for l in open("ids/test_ids.txt") if l.strip()]

# ---- Choose a few random test samples ----
random.seed(42)
TEST_IDS = random.sample(TEST_IDS, min(5, len(TEST_IDS)))

# ---- Preprocessing ----
tfms = A.Compose([
    A.Resize(1024, 1024),
    ToTensorV2()
])

# ---- Load pretrained U-Net (ImageNet encoder) ----
model = Unet(
    encoder_name="resnet34",      # simple, fast backbone
    encoder_weights="imagenet",   # pretrained
    in_channels=3,
    classes=1
)
model.eval()  # no training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---- Metric functions ----
def iou(pred, targ, eps=1e-6):
    inter = (pred & targ).sum()
    union = (pred | targ).sum()
    return (inter + eps) / (union + eps)

def dice(pred, targ, eps=1e-6):
    inter = (pred & targ).sum()
    return (2 * inter + eps) / (pred.sum() + targ.sum() + eps)

ious, dices = [], []

# ---- Loop through a few samples ----
for sid in tqdm(TEST_IDS, desc="Running dummy test"):
    # find image path
    ext = ".png"
    path = os.path.join(IMG_DIR, sid + ext)
    if os.path.exists(path):
        img_path = path
    else:
        print(f"⚠️ No image for {sid}")
        continue

    mask_path = os.path.join(MASK_DIR, sid + "_mask.npy")
    if not os.path.exists(mask_path):
        print(f"⚠️ No mask for {sid}")
        continue

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = np.load(mask_path).astype(np.uint8)

    aug = tfms(image=img, mask=mask)
    x = aug["image"].float().unsqueeze(0) / 255.0
    y = aug["mask"].unsqueeze(0)

    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = model(x)
        pred = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8)

    i = iou(pred, y.cpu().numpy())
    d = dice(pred, y.cpu().numpy())
    ious.append(i)
    dices.append(d)

    # Optional: visualize one example
    if sid == TEST_IDS[0]:
        import matplotlib.pyplot as plt
        img_disp = (x[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        pred_mask = pred[0,0]*255
        overlay = img_disp.copy()
        overlay[pred_mask>0] = (0.4*overlay[pred_mask>0] + 0.6*np.array([0,255,0])).astype(np.uint8)
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.imshow(img_disp); plt.title("Image"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(overlay); plt.title("Predicted mask overlay"); plt.axis("off")
        plt.show()

print("------ Dummy U-Net test results ------")
print(f"Mean IoU : {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
