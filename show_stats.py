import torch
from segmentation_models_pytorch import Unet

from myUnet import SmallResUNet
from overfit_method import tfms_random_crop_normalized, tfms_normalized, combined_loss, \
    tfms_random_crop_normalized_encoder_vals, tfms_normalized_encoder_vals, tfms_val_normalized_encoder_vals, \
    tfms_building_focus_train, evaluate_iou_split
from overfit_trials import get_train_test_loaders
from stats import get_stats, show_stats

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
        encoder_name="resnet34",  #TODO motivation why we chose this as starter
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
state = torch.load("unet/unet_train_test_split_found_building_focus_p_09_lr_005.pt", map_location=device)
model.load_state_dict(state)

print(model)

_, test_loader = get_train_test_loaders(tfms_building_focus_train, tfms_val_normalized_encoder_vals)
evaluate_iou_split(model, test_loader, device)
# stats = get_stats(model, train_loader, combined_loss)
# show_stats(stats, "unet_train_test_split_found_lr_0003_x_axis")
