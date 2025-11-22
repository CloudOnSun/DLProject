import os

import keras
import numpy as np
import torch
from matplotlib import pyplot as plt
from segmentation_models_pytorch import Unet
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from torch.utils.data import DataLoader
from tqdm import tqdm

from overfit_method import main, read_ids, OverfitDataset, init_kaiming_for_conv, combined_loss, tfms_basic, \
    tfms_random_crop, tfms_random_crop_normalized, tfms_normalized


def get_overfit_test_loaders(train_tfms, test_tfms):
    overfit_ids = read_ids("ids/overfit_ids.txt")
    print("Loaded", len(overfit_ids), "overfit samples")
    if len(overfit_ids) == 0:
        raise RuntimeError("overfit_ids.txt is empty. Check how you generated it.")

    # test split
    test_ids = read_ids("ids/test_ids_subsample.txt")
    print("Loaded", len(test_ids), "test samples")
    if len(test_ids) == 0:
        raise RuntimeError("test_ids_subsample.txt is empty.")

    train_ds = OverfitDataset(overfit_ids, tfms=train_tfms)
    test_ds = OverfitDataset(test_ids, tfms=test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, test_loader

def get_valid_test_loaders(train_tfms, test_tfms):
    overfit_ids = read_ids("ids/val_ids_subsample.txt")
    print("Loaded", len(overfit_ids), "overfit samples")
    if len(overfit_ids) == 0:
        raise RuntimeError("val_ids_subsample.txt is empty. Check how you generated it.")

    # test split
    test_ids = read_ids("ids/test_ids_subsample.txt")
    print("Loaded", len(test_ids), "test samples")
    if len(test_ids) == 0:
        raise RuntimeError("test_ids_subsample.txt is empty.")

    train_ds = OverfitDataset(overfit_ids, tfms=train_tfms)
    test_ds = OverfitDataset(test_ids, tfms=test_tfms)

    batch_size = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, test_loader

def get_train_test_loaders(train_tfms, test_tfms):
    overfit_ids = read_ids("ids/train_ids_subsample.txt")
    print("Loaded", len(overfit_ids), "train samples")
    if len(overfit_ids) == 0:
        raise RuntimeError("train_ids_subsample.txt is empty. Check how you generated it.")

    # test split
    test_ids = read_ids("ids/test_ids_subsample.txt")
    print("Loaded", len(test_ids), "test samples")
    if len(test_ids) == 0:
        raise RuntimeError("test_ids_subsample.txt is empty.")

    train_ds = OverfitDataset(overfit_ids, tfms=train_tfms)
    test_ds = OverfitDataset(test_ids, tfms=test_tfms)

    batch_size = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, test_loader

def trial_basic_for_overfit(train_loader, test_loader, dataset_split, trial_name, trial_id):
    print("Running trial " + trial_name + " on dataset " + dataset_split + " id " + trial_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    main(train_loader=train_loader,
         test_loader=test_loader,
         model=model,
         optimizer=optimizer,
         epochs=epochs,
         loss_func=combined_loss,
         model_file="unet/unet_" + dataset_split + "_" + trial_name + ".pt",
         train_loss_file="plots/" + dataset_split + "/" + trial_name + "/train_loss_curve" + trial_id +".png",
         test_loss_file="plots/" + dataset_split + "/" + trial_name + "/test_loss_curve" + trial_id +".png",
         iou_file="plots/" + dataset_split + "/" + trial_name + "/iou_curve" + trial_id +".png",
         device=device)


def trial_basic(train_loader, test_loader, dataset_split, trial_name, trial_id):
    print("Running trial " + trial_name + " on dataset " + dataset_split + " id " + trial_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 100

    main(train_loader=train_loader,
         test_loader=test_loader,
         model=model,
         optimizer=optimizer,
         epochs=epochs,
         loss_func=combined_loss,
         model_file="unet/unet_" + dataset_split + "_" + trial_name + ".pt",
         train_loss_file="plots/" + dataset_split + "/" + trial_name + "/train_loss_curve" + trial_id +".png",
         test_loss_file="plots/" + dataset_split + "/" + trial_name + "/test_loss_curve" + trial_id +".png",
         iou_file="plots/" + dataset_split + "/" + trial_name + "/iou_curve" + trial_id +".png",
         device=device)

def trial_lr_decay(train_loader, test_loader, dataset_split, trial_name, trial_id):
    print("Running trial " + trial_name + " on dataset " + dataset_split + " id " + trial_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.96  # multiply LR by
    )
    epochs = 50

    main(train_loader=train_loader,
         test_loader=test_loader,
         model=model,
         optimizer=optimizer,
         epochs=epochs,
         loss_func=combined_loss,
         model_file="unet/unet_" + dataset_split + "_" + trial_name + ".pt",
         train_loss_file="plots/" + dataset_split + "/" + trial_name + "/train_loss_curve" + trial_id +".png",
         test_loss_file="plots/" + dataset_split + "/" + trial_name + "/test_loss_curve" + trial_id +".png",
         iou_file="plots/" + dataset_split + "/" + trial_name + "/iou_curve" + trial_id +".png",
         device=device,
         scheduler=scheduler)


def trial_with_weight_init(train_loader, test_loader, dataset_split, trial_name, trial_id):
    print("Running trial " + trial_name + " on dataset " + dataset_split + " id " + trial_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # apply the kaiming init for decoder only
    model.decoder.apply(init_kaiming_for_conv)
    model.segmentation_head.apply(init_kaiming_for_conv)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 20

    main(train_loader=train_loader,
         test_loader=test_loader,
         model=model,
         optimizer=optimizer,
         epochs=epochs,
         loss_func=combined_loss,
         model_file="unet/unet_" + dataset_split + "_" + trial_name + ".pt",
         train_loss_file="plots/" + dataset_split + "/" + trial_name + "/train_loss_curve" + trial_id +".png",
         test_loss_file="plots/" + dataset_split + "/" + trial_name + "/test_loss_curve" + trial_id +".png",
         iou_file="plots/" + dataset_split + "/" + trial_name + "/iou_curve" + trial_id +".png",
         device=device)

def trial_with_find_lr(train_loader, test_loader, dataset_split, trial_name, trial_id):
    print("Running trial " + trial_name + " on dataset " + dataset_split + " id " + trial_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    epochs = 1
    t = 1
    T = 147
    lr_list = np.logspace(-6, 0, 147)

    def lr_finder(t, T, lr):
        return lr_list[t - 1]

    # apply the kaiming init for decoder only
    model.decoder.apply(init_kaiming_for_conv)
    model.segmentation_head.apply(init_kaiming_for_conv)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss_func=combined_loss

    train_loss_history = []

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        # ----- train on overfit subset -----
        model.train()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            optimizer.param_groups[0]['lr'] = lr_finder(t, T, lr=optimizer.param_groups[0]['lr'])
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_func(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss_history.append(loss.item())
            t += 1


    plt.figure()
    plt.plot(lr_list, train_loss_history)
    plt.xscale('log')  # Use log scale on x-axis
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')

    save_path = "plots/" + dataset_split + "/" + trial_name + "/learning_rates" + trial_id +".png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def trial_with_weight_decay(train_loader, test_loader, dataset_split, trial_name, trial_id):
    print("Running trial " + trial_name + " on dataset " + dataset_split + " id " + trial_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    epochs = 20

    main(train_loader=train_loader,
         test_loader=test_loader,
         model=model,
         optimizer=optimizer,
         epochs=epochs,
         loss_func=combined_loss,
         model_file="unet/unet_" + dataset_split + "_" + trial_name + ".pt",
         train_loss_file="plots/" + dataset_split + "/" + trial_name + "/train_loss_curve" + trial_id +".png",
         test_loss_file="plots/" + dataset_split + "/" + trial_name + "/test_loss_curve" + trial_id +".png",
         iou_file="plots/" + dataset_split + "/" + trial_name + "/iou_curve" + trial_id +".png",
         device=device)

if __name__ == "__main__":
    train_loader, test_loader = get_train_test_loaders(tfms_random_crop_normalized, tfms_normalized)
    trial_lr_decay(train_loader, test_loader, "train_test_split", "normalize_data", "9")