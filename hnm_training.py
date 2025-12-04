import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from segmentation_models_pytorch import Unet
from tqdm import tqdm

from method import (
    tfms_val_normalized_encoder_vals,
    combined_boundary_hnm_loss,
    tfms_building_focus_train,
    main
)
from trials import get_train_test_loaders


def run_phase2_hnm(
    phase1_model_path,
    dataset_split="train_test_split",
    save_name="hnm_model",
    epochs=10,
    lr=0.0005,
    alpha=4.0,
    neg_ratio=0.25,
    hnm_weight=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_train_test_loaders(
        tfms_building_focus_train,
        tfms_val_normalized_encoder_vals
    )
    print("\nBuilding U-Net model...")
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,       # DO NOT load imagenet weights again
        in_channels=3,
        classes=1
    ).to(device)

    print("Loading Phase-1 checkpoint:", phase1_model_path)
    state = torch.load(phase1_model_path, map_location=device)
    model.load_state_dict(state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    loss_func = lambda logits, targets: combined_boundary_hnm_loss(
        logits,
        targets,
        alpha=alpha,
        neg_ratio=neg_ratio,
        hnm_weight=hnm_weight
    )

    model_file = f"unet/{save_name}.pt"
    train_plot = f"plots/{dataset_split}/{save_name}/train_loss_phase2.png"
    test_plot  = f"plots/{dataset_split}/{save_name}/test_loss_phase2.png"
    iou_plot   = f"plots/{dataset_split}/{save_name}/iou_phase2.png"

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname(train_plot), exist_ok=True)

    print("\n=== Starting Phase-2 Fine-Tuning with HNM ===")

    main(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        loss_func=loss_func,
        model_file=model_file,
        train_loss_file=train_plot,
        test_loss_file=test_plot,
        iou_file=iou_plot,
        device=device,
        scheduler=scheduler
    )

    print("\nPHASE 2 COMPLETE!")
    print("Saved:", model_file)


def trial_with_find_lr_phase2_hnm(
    train_loader,
    dataset_split,
    trial_name,
    trial_id,
    phase1_model_path,
    alpha=4.0,
    neg_ratio=0.25,
    hnm_weight=0.5
):
    """
    LR Finder for Phase 2 HNM training.
    Loads Phase 1 model, then sweeps LR from 1e-6 to 1.0.
    """


    print(f"Running Phase 2 LR finder {trial_name} on dataset {dataset_split} id {trial_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------------------------------
    # 1. Build SAME model architecture
    # -------------------------------------------------------------
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # DO NOT reload imagenet
        in_channels=3,
        classes=1
    ).to(device)

    # -------------------------------------------------------------
    # 2. Load Phase 1 checkpoint
    # -------------------------------------------------------------
    print(f"Loading Phase 1 model from: {phase1_model_path}")
    state = torch.load(phase1_model_path, map_location=device)
    model.load_state_dict(state)

    # -------------------------------------------------------------
    # 3. LR sweep setup (logspace like your previous version)
    # -------------------------------------------------------------
    epochs = 1
    T = 147
    t = 1
    lr_list = np.logspace(-6, 0, T)  # 1e-6 → 1

    def lr_finder(t, T, lr):
        return lr_list[t - 1]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

    # -------------------------------------------------------------
    # 4. The HNM + boundary loss
    # -------------------------------------------------------------
    loss_func = lambda logits, targets: combined_boundary_hnm_loss(
        logits,
        targets,
        alpha=alpha,
        neg_ratio=neg_ratio,
        hnm_weight=hnm_weight
    )

    train_loss_history = []

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -------------------------------------------------------------
    # 5. LR Finder Loop
    # -------------------------------------------------------------
    model.train()

    for epoch in range(1, epochs + 1):

        for x, y in tqdm(train_loader, desc=f"[LR Finder Phase 2] Epoch {epoch}/{epochs}"):

            # update LR
            optimizer.param_groups[0]['lr'] = lr_finder(t, T, optimizer.param_groups[0]['lr'])

            # move data
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

            if t > T:
                break

        if t > T:
            break

    # -------------------------------------------------------------
    # 6. Plot LR vs Loss
    # -------------------------------------------------------------
    plt.figure()
    plt.plot(lr_list, train_loss_history)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title("LR Finder (Phase 2 + HNM)")

    save_dir = f"plots/{dataset_split}/{trial_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/lr_finder_phase2_{trial_id}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved LR finder plot to {save_path}")


if __name__ == "__main__":
    # phase1_model_path = "unet/unet_train_test_split_found_building_focus_p_09_lr_005.pt"
    #
    # run_phase2_hnm(
    #     phase1_model_path=phase1_model_path,
    #     dataset_split="train_test_split",
    #     save_name="boundary_hnm_refined",
    #     epochs=10,
    #     lr=0.0005,
    #     alpha=4.0,
    #     neg_ratio=0.25,
    #     hnm_weight=0.5
    # )

    train_loader, _ = get_train_test_loaders(
        tfms_building_focus_train,
        tfms_val_normalized_encoder_vals
    )

    trial_with_find_lr_phase2_hnm(
        train_loader=train_loader,
        dataset_split="train_test_split",
        trial_name="phase2_hnm",
        trial_id="01",
        phase1_model_path="unet/unet_train_test_split_found_building_focus_p_09_lr_005.pt",
        alpha=4.0,
        neg_ratio=0.25,
        hnm_weight=0.5
    )