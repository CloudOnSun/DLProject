import os
import csv
import cv2
import numpy as np
import torch
from segmentation_models_pytorch import Unet

from overfit_method import (
    read_ids,
    tfms_val_normalized_encoder_vals,
    iou_score,
)
from plot_utils import predict_and_show

# ---------------- config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "unet/unet_train_test_split_found_lr_0003.pt"
test_ids_file = "ids/test_ids_subsample.txt"

IMG_DIR = "data/images"
MASK_DIR = "data/masks"

pred_out_root = "predictions/all_test_predictions"
os.makedirs(pred_out_root, exist_ok=True)

csv_out = "metrics/iou_per_test_image.csv"
os.makedirs(os.path.dirname(csv_out), exist_ok=True)

# ------------- load model ---------------
model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,     # IMPORTANT
    in_channels=3,
    classes=1,
).to(device)

state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# ------------- load test IDs ------------
test_ids = read_ids(test_ids_file)
print(f"Loaded {len(test_ids)} test images")

# ------------- loop over test set -------
results = []

with torch.no_grad():
    for sid in test_ids:
        print(f"Processing {sid} ...")

        # ----------------- load raw image + mask -----------------
        img_path = os.path.join(IMG_DIR, sid + ".png")
        mask_path = os.path.join(MASK_DIR, sid + "_mask.npy")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path).astype(np.uint8)

        # ----------------- apply validation transforms ------------
        augmented = tfms_val_normalized_encoder_vals(image=img, mask=mask)
        x = augmented["image"].unsqueeze(0).to(device)          # [1,3,H,W]
        y = augmented["mask"].float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

        # ----------------- compute logits -------------------------
        logits = model(x)

        # ----------------- compute IoU ----------------------------
        iou = iou_score(logits, y)
        results.append((sid, iou))

        # ----------------- save prediction plot -------------------
        save_path = os.path.join(pred_out_root, f"{sid}_prediction.png")
        predict_and_show(
            model=model,
            tile_id=sid,
            tfms=tfms_val_normalized_encoder_vals,
            device=device,
            save_path=save_path
        )

        print(f"    IoU={iou:.4f}  → saved {save_path}")

# ------------- sort + write CSV ----------
results_sorted = sorted(results, key=lambda t: t[1])  # ascending = worst first

with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tile_id", "iou"])
    for sid, iou in results_sorted:
        writer.writerow([sid, f"{iou:.6f}"])

print("\nSaved CSV:", csv_out)

print("\nWorst 5 predictions:")
for sid, iou in results_sorted[:5]:
    print(f"  {sid}: {iou:.4f}")

print("\nBest 5 predictions:")
for sid, iou in results_sorted[-5:]:
    print(f"  {sid}: {iou:.4f}")
