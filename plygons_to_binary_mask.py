# make_all_masks.py
import os
import json
import numpy as np
from shapely import wkt
from shapely.geometry import box
from rasterio.features import rasterize
from affine import Affine
from tqdm import tqdm

LABEL_DIR = "data/labels"
MASK_DIR  = "data/masks"
os.makedirs(MASK_DIR, exist_ok=True)

def polygons_from_xy(json_path):
    """Extract building polygons (pixel coords) and image size from an xBD/xView2 label JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    width  = int(data["metadata"]["width"])
    height = int(data["metadata"]["height"])

    polys = []
    for feat in data["features"]["xy"]:
        if feat.get("properties", {}).get("feature_type") != "building":
            continue
        try:
            geom = wkt.loads(feat["wkt"])
            polys.append(geom)
        except Exception as e:
            print(f"Warning: could not parse WKT in {json_path}: {e}")
    return polys, width, height

def rasterize_buildings(polys, width, height):
    """Rasterize list of polygons into a binary mask."""
    if len(polys) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    img_bounds = box(0, 0, width, height)
    clipped = []
    for g in polys:
        inter = g.intersection(img_bounds)
        if not inter.is_empty:
            clipped.append((inter, 1))

    mask = rasterize(
        shapes=clipped,
        out_shape=(height, width),
        fill=0,
        dtype="uint8",
        transform=Affine.identity(),
        all_touched=False,
    )
    return mask

def main():
    json_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".json")]
    for jf in tqdm(json_files, desc="Converting JSON → mask"):
        json_path = os.path.join(LABEL_DIR, jf)
        stem = os.path.splitext(jf)[0]
        out_path = os.path.join(MASK_DIR, stem + "_mask.npy")

        polys, width, height = polygons_from_xy(json_path)
        mask = rasterize_buildings(polys, width, height)
        np.save(out_path, mask)

    print(f"Done! Masks saved in {MASK_DIR}/")

main()
