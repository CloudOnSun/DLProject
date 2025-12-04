import torch
from torch.utils.data import DataLoader

from method import read_ids, BuildingDataset, tfms_basic


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    n = 0
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)

    for imgs, _ in loader:
        # imgs shape: (B, C, H, W)
        imgs = imgs.float() / 255.0  # convert to [0,1] before computing stats
        b, c, h, w = imgs.shape

        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_squared_sum += (imgs ** 2).sum(dim=[0, 2, 3])
        n += b * h * w

    mean = channel_sum / n
    std = torch.sqrt(channel_squared_sum / n - mean ** 2)
    return mean, std


train_ids = read_ids("ids/train_ids_subsample.txt")
print("Loaded", len(train_ids), "train samples")
if len(train_ids) == 0:
    raise RuntimeError("train_ids_subsample.txt is empty. Check how you generated it.")

train_ds = BuildingDataset(train_ids, tfms=tfms_basic)
mean, std = compute_mean_std(train_ds)
print("Mean:", mean)
print("Std:", std)
