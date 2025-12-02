import random
import numpy as np
import albumentations as A

class RandomBuildingCrop(A.DualTransform):
    """
    Albumentations-compatible crop focused on buildings (mask > 0).

    - With probability p_focus: choose a random foreground pixel
      and crop around it (clamped to image bounds).
    - Otherwise: crop randomly anywhere.

    Returns:
      {"image": cropped_image, "mask": cropped_mask}
    """

    def __init__(self, height, width, p_focus=0.5, always_apply=False, p=1.0):
        super(RandomBuildingCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.p_focus = p_focus

    def apply(self, img, x_min=0, y_min=0, **params):
        return img[y_min:y_min+self.height, x_min:x_min+self.width]

    def apply_to_mask(self, mask, x_min=0, y_min=0, **params):
        return mask[y_min:y_min+self.height, x_min:x_min+self.width]

    def get_params_dependent_on_targets(self, params):
        """
        Compute the crop coordinates. Must return a dict with keys used in apply().
        """
        img = params["image"]
        mask = params["mask"]

        h, w = mask.shape[:2]
        ch, cw = self.height, self.width

        # fallback for small images
        if h <= ch or w <= cw:
            y_min = max(0, (h - ch) // 2)
            x_min = max(0, (w - cw) // 2)
            return {"x_min": x_min, "y_min": y_min}

        use_focus = (random.random() < self.p_focus) and (mask.max() > 0)

        if use_focus:
            # pick a random foreground pixel
            ys, xs = np.where(mask > 0)
            idx = np.random.randint(len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])

            # center crop around pixel, but clamp to valid range
            y_min = np.clip(cy - ch // 2, 0, h - ch)
            x_min = np.clip(cx - cw // 2, 0, w - cw)
        else:
            # uniform random crop
            y_min = np.random.randint(0, h - ch + 1)
            x_min = np.random.randint(0, w - cw + 1)

        return {"x_min": x_min, "y_min": y_min}

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        return ("height", "width", "p_focus")
