import torch
from segmentation_models_pytorch import Unet
from overfit_trials import get_train_test_loaders
from overfit_method import (
    gradcam_decoder_for_loader,
    tfms_normalized_encoder_vals,
    tfms_val_normalized_encoder_vals,
    tfms_random_crop_normalized_encoder_vals
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test loader
    _, test_loader = get_train_test_loaders(
        tfms_random_crop_normalized_encoder_vals,
        tfms_val_normalized_encoder_vals
    )

    # Recreate model architecture
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # prevent loading new imagenet weights
        in_channels=3,
        classes=1
    ).to(device)

    # Load trained weights
    model_path = "unet/unet_train_test_split_no_random_crop_100_epoch.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run Grad-CAM on the saved model
    gradcam_decoder_for_loader(
        model=model,
        loader=test_loader,
        device=device,
        save_dir="plots/train_test_split/no_random_crop_100_epoch/gradcam_decoder_loaded",
        denorm_mode="imagenet",
        max_samples=10
    )
