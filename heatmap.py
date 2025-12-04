import torch
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader

from trials import get_train_test_loaders
from method import (
    gradcam_decoder_for_loader,
    tfms_normalized_encoder_vals,
    tfms_val_normalized_encoder_vals,
    tfms_random_crop_normalized_encoder_vals, BuildingDataset
)

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    ids = ["hurricane-florence_00000001_pre_disaster"]
    ds = BuildingDataset(ids, tfms=tfms_val_normalized_encoder_vals)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # # Load test loader
    # _, test_loader = get_train_test_loaders(
    #     tfms_random_crop_normalized_encoder_vals,
    #     tfms_val_normalized_encoder_vals
    # )

    # Recreate model architecture
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # prevent loading new imagenet weights
        in_channels=3,
        classes=1
    ).to(device)

    # Load trained weights
    model_path = "unet/unet_train_test_split_found_lr_0003.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run Grad-CAM on the saved model
    gradcam_decoder_for_loader(
        model=model,
        loader=loader,
        device=device,
        save_dir="plots/predictions/" + ids[0] + "gradcam",
        denorm_mode="imagenet",
        max_samples=1
    )
