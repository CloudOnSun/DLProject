import os
import torch
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm

from method import (
    BuildingDataset,
    read_ids,
    tfms_val_normalized_encoder_vals,
)
from stats import save_error_maps_per_image


# ------------------- CONFIG -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_path = "unet/unet_train_test_split_found_building_focus_p_09_lr_005.pt"
test_ids_file = "ids/test_ids_subsample.txt"

save_dir = "plots/error_maps_per_image_focus_building"
os.makedirs(save_dir, exist_ok=True)


# ------------------- LOAD MODEL -------------------
model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,      # IMPORTANT: do NOT reload imagenet weights
    in_channels=3,
    classes=1
).to(device)

state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

print("Loaded model from:", model_path)


# ------------------- LOAD TEST DATA -------------------
test_ids = read_ids(test_ids_file)
print(f"Loaded {len(test_ids)} test samples")

test_ds = BuildingDataset(test_ids, tfms=tfms_val_normalized_encoder_vals)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


# ------------------- RUN ERROR MAPS -------------------
results = save_error_maps_per_image(
    model=model,
    loader=test_loader,
    device=device,
    save_dir=save_dir,
    threshold=0.5,
)

print("Saved all error maps to:", save_dir)
print("Example entry:", results[0])
