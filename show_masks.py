import os
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR = "data/images"      # directory containing RGB images
MASK_DIR  = "data/masks"       # where *_mask.npy are saved
OUT_DIR   = "data/overlays_overfit"    # folder to save overlay images
ID_FILE   = "ids/overfit_ids.txt"  # file containing selected image IDs

os.makedirs(OUT_DIR, exist_ok=True)

def load_ids(path):
    """Load one ID per line, strip whitespace."""
    with open(path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

def visualize_single(stem, save=True):
    """
    stem = filename without extension.
    Example: 'hurricane-florence_00000123'
    """

    # ---- Load image (adjust extension: .png/.jpg/.tif etc.) ----
    img_path = os.path.join(IMAGE_DIR, stem + ".png")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    img = plt.imread(img_path)

    # ---- Load mask ----
    mask_path = os.path.join(MASK_DIR, stem + "_mask.npy")
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return

    mask = np.load(mask_path)

    # ---- Overlay ----
    rgba_mask = np.zeros((*mask.shape, 4))
    rgba_mask[mask == 1] = [1, 0, 0, 0.6]  # Red, alpha 0.6
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    # plt.imshow(np.ma.masked_where(mask == 0, mask),
    #            cmap="Reds", alpha=0.45)
    plt.imshow(rgba_mask)

    plt.axis("off")
    plt.title(stem)

    # ---- Save or show ----
    if save:
        out_path = os.path.join(OUT_DIR, stem + "_overlay.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    plt.close()  # close to avoid memory buildup

def main():
    stems = load_ids(ID_FILE)
    print(f"Loaded {len(stems)} IDs from {ID_FILE}")

    for s in stems:
        visualize_single(s, save=True)

    print("Done generating overlays.")

if __name__ == "__main__":
    main()
