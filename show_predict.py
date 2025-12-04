import torch
from segmentation_models_pytorch import Unet
from method import tfms_basic, tfms_normalized, tfms_normalized_encoder_vals, iou_score, tfms_val_normalized_encoder_vals

from plot_utils import predict_and_show

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
        encoder_name="resnet34",  #TODO motivation why we chose this as starter
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
state = torch.load("unet/unet_train_test_split_found_building_focus_p_09_lr_005.pt", map_location=device)
model.load_state_dict(state)
model.eval()

id = "hurricane-florence_00000015_pre_disaster"
predict_and_show(model, tile_id="hurricane-florence_00000015_pre_disaster", tfms=tfms_val_normalized_encoder_vals, device=device,
                 save_path="predictions/" + id + "_predict2.png")