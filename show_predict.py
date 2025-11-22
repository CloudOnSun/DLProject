import torch
from segmentation_models_pytorch import Unet
from overfit_method import tfms_basic, tfms_normalized

from plot_utils import predict_and_show

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
        encoder_name="resnet34",  #TODO motivation why we chose this as starter
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
state = torch.load("unet/unet_train_test_split_normalize_data.pt", map_location=device)
model.load_state_dict(state)
model.eval()

id = "hurricane-florence_00000027_pre_disaster"
predict_and_show(model, tile_id="hurricane-florence_00000027_pre_disaster", tfms=tfms_normalized, device=device,
                 save_path="predictions/" + id + "_predict9.png")