import torch
from segmentation_models_pytorch import Unet

from myUnet import SmallResUNet
from overfit_method import tfms_random_crop_normalized, tfms_normalized, combined_loss, \
    tfms_random_crop_normalized_encoder_vals, tfms_normalized_encoder_vals, tfms_val_normalized_encoder_vals
from overfit_trials import get_train_test_loaders
from stats import get_stats, show_stats, aggregate_error_stats_and_maps

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
        encoder_name="resnet34",  #TODO motivation why we chose this as starter
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
state = torch.load("unet/unet_train_test_split_found_lr_0003.pt", map_location=device)
model.load_state_dict(state)

print(model)

train_loader, test_loader = get_train_test_loaders(tfms_random_crop_normalized_encoder_vals, tfms_val_normalized_encoder_vals)
# stats = get_stats(model, train_loader, combined_loss)
# show_stats(stats, "unet_train_test_split_found_lr_0003")
aggregate_error_stats_and_maps(model, test_loader, device, save_path="stats/heat_map/unet_train_test_split_found_lr_0003")