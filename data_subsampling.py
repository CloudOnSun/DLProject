import os
import json
import csv
import random
from collections import defaultdict

from shapely import wkt

# ---------------- CONFIG ----------------
LABEL_DIR = "data/labels"              # where your *.json files live
OUT_CSV_SELECTED = "selected_images_splits_60.csv"
OUT_CSV_ALL      = "all_images_stats_60.csv"

# only use pre-disaster labels
def is_pre_disaster(filename: str) -> bool:
    # adjust if your naming scheme differs
    return filename.endswith("_pre_disaster.json")

KEEP_FRACTION = 0.6   # e.g. keep 30% of images overall (tune this!)

# ratio bins: [edge0, edge1), [edge1, edge2), ...
# you can later update these after inspecting the plots
RATIO_BIN_EDGES = [-1, 0.001010, 0.012155, 0.075244, 0.631568]
RATIO_BIN_NAMES = ["very_low", "low", "medium", "high"]

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

RANDOM_SEED = 1337
# ----------------------------------------


def assign_ratio_bin(ratio: float) -> str:
    """
    Assign a building_ratio value to one of the named bins
    defined by RATIO_BIN_EDGES and RATIO_BIN_NAMES.
    """
    for i in range(len(RATIO_BIN_EDGES) - 1):
        lo = RATIO_BIN_EDGES[i]
        hi = RATIO_BIN_EDGES[i + 1]
        if lo < ratio <= hi:
            return RATIO_BIN_NAMES[i]
    # fallback (should almost never happen)
    return RATIO_BIN_NAMES[-1]


def compute_stats_from_json(json_path: str):
    """
    Parse one xView2/xBD-style JSON label file and compute:
      - tile_id
      - width, height
      - number of building instances
      - building_ratio = total building polygon area / (W*H)
      - ratio_bin
      - inferred image_name and mask_name
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data["metadata"]
    width  = int(meta["width"])
    height = int(meta["height"])
    tile_id = os.path.splitext(os.path.basename(json_path))[0]  # e.g. 'guatemala-volcano_00000006_pre_disaster'

    total_area = 0.0
    n_buildings = 0

    for feat in data["features"]["xy"]:
        if feat.get("properties", {}).get("feature_type") != "building":
            continue
        poly = wkt.loads(feat["wkt"])
        n_buildings += 1
        total_area += poly.area   # in pixel units

    building_ratio = total_area / (width * height) if width > 0 and height > 0 else 0.0
    ratio_bin = assign_ratio_bin(building_ratio)

    return {
        "tile_id": tile_id,
        "json_path": json_path,
        "width": width,
        "height": height,
        "n_buildings": n_buildings,
        "building_ratio": building_ratio,
        "ratio_bin": ratio_bin,
        # convenience: how you'd name image/mask files
        "image_name": tile_id + ".png",      # adapt extension if needed
        "mask_name": tile_id + "_mask.npy",  # if you follow your previous convention
    }


def main():
    random.seed(RANDOM_SEED)

    # ---------------- 1. Collect all pre-disaster JSONs ----------------
    json_files = [
        os.path.join(LABEL_DIR, f)
        for f in os.listdir(LABEL_DIR)
        if f.endswith(".json") and is_pre_disaster(f)
    ]
    print(f"Found {len(json_files)} pre-disaster label files.")

    # ---------------- 2. Compute stats for every image ----------------
    all_records = []
    for jp in json_files:
        try:
            rec = compute_stats_from_json(jp)
            all_records.append(rec)
        except Exception as e:
            print(f"[WARN] Failed on {jp}: {e}")

    print(f"Computed stats for {len(all_records)} images.")

    # ---------------- 3. Write pre-subsampling stats to CSV ----------------
    fieldnames_all = [
        "tile_id",
        "json_path",
        "image_name",
        "mask_name",
        "width",
        "height",
        "n_buildings",
        "building_ratio",
        "ratio_bin",
    ]

    with open(OUT_CSV_ALL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_all)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)

    print(f"Wrote pre-subsampling stats for {len(all_records)} images to {OUT_CSV_ALL}")

    # ---------------- 4. Group by ratio_bin ----------------
    bins = defaultdict(list)
    for idx, rec in enumerate(all_records):
        bins[rec["ratio_bin"]].append(idx)

    print("Bin distribution (before subsampling):")
    for bname in RATIO_BIN_NAMES:
        print(f"  {bname}: {len(bins[bname])} images")

    # ---------------- 5. Subsample within each bin ----------------
    selected_indices = []
    for bname in RATIO_BIN_NAMES:
        idxs = bins[bname]
        if not idxs:
            continue

        k = max(1, int(round(KEEP_FRACTION * len(idxs))))
        k = min(k, len(idxs))
        chosen = random.sample(idxs, k)
        selected_indices.extend(chosen)

        print(f"Selected {k}/{len(idxs)} images from bin '{bname}'.")

    selected_indices = sorted(selected_indices)
    selected_records = [all_records[i] for i in selected_indices]
    print(f"Total selected images: {len(selected_records)}")

    # ---------------- 6. Stratified train/val/test split by ratio_bin ----------------
    bins_sel = defaultdict(list)
    for rec in selected_records:
        bins_sel[rec["ratio_bin"]].append(rec)

    for bname in RATIO_BIN_NAMES:
        recs = bins_sel[bname]
        if not recs:
            continue
        random.shuffle(recs)

        n = len(recs)
        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)
        # ensure all images are used
        n_test  = n - n_train - n_val

        train_recs = recs[:n_train]
        val_recs   = recs[n_train:n_train + n_val]
        test_recs  = recs[n_train + n_val:]

        for r in train_recs:
            r["split"] = "train"
        for r in val_recs:
            r["split"] = "val"
        for r in test_recs:
            r["split"] = "test"

        print(
            f"Bin '{bname}': "
            f"{len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test"
        )

    # ---------------- 7. Write selected + splits to CSV ----------------
    fieldnames_selected = [
        "tile_id",
        "json_path",
        "image_name",
        "mask_name",
        "width",
        "height",
        "n_buildings",
        "building_ratio",
        "ratio_bin",
        "split",
    ]

    with open(OUT_CSV_SELECTED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_selected)
        writer.writeheader()
        for rec in selected_records:
            # just in case, but all should have split set
            if "split" not in rec:
                rec["split"] = "unsplit"
            writer.writerow(rec)

    print(f"Wrote {len(selected_records)} rows to {OUT_CSV_SELECTED}")

    # ---------------- 8. Write train/val/test ID text files ----------------
    train_ids = [rec["tile_id"] for rec in selected_records if rec["split"] == "train"]
    val_ids   = [rec["tile_id"] for rec in selected_records if rec["split"] == "val"]
    test_ids  = [rec["tile_id"] for rec in selected_records if rec["split"] == "test"]

    def write_list(path, items):
        with open(path, "w") as f:
            for item in items:
                f.write(item + "\n")
        print(f"Wrote {len(items)} IDs to {path}")

    write_list("ids/train_ids_subsample_60.txt", train_ids)
    write_list("ids/val_ids_subsample_60.txt", val_ids)
    write_list("ids/test_ids_subsample_60.txt", test_ids)


if __name__ == "__main__":
    main()
