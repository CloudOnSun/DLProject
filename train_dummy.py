import os, cv2, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segmentation_models_pytorch import Unet

IMG_DIR  = "data/images"
MASK_DIR = "data/masks"

def read_ids(path):
    return [x.strip() for x in open(path) if x.strip()]

TRAIN_IDS = read_ids("ids/train_ids.txt")
VAL_IDS   = read_ids("ids/val_ids.txt")
TEST_IDS  = read_ids("ids/test_ids.txt")

ext = ".png"



class XBD(Dataset):
    def __init__(self, ids):
        self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        sid = self.ids[i]
        img_path  = os.path.join(IMG_DIR, sid + ".png")
        mask_path = os.path.join(MASK_DIR, sid + "_mask.npy")

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path).astype(np.uint8)

        # ensure sizes match, no resizing/cropping
        if (img.shape[0] != mask.shape[0]) or (img.shape[1] != mask.shape[1]):
            raise ValueError(
                f"Size mismatch for {sid}: image {img.shape[:2]} vs mask {mask.shape}"
            )

        x = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # CxHxW
        y = torch.from_numpy(mask).unsqueeze(0).float()           # 1xHxW
        return x, y

def make_loader(ids, bs=32, shuffle=False):
    return DataLoader(XBD(ids), batch_size=bs, shuffle=shuffle, num_workers=0)

# dataloaders (bs=1 to keep it truly "no processing")
train_dl = make_loader(TRAIN_IDS, bs=2, shuffle=True)
val_dl   = make_loader(VAL_IDS,   bs=2, shuffle=False)
test_dl  = make_loader(TEST_IDS,  bs=2, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",  # pretrained encoder
    in_channels=3,
    classes=1
).to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

@torch.no_grad()
def eval_iou(dloader):
    model.eval()
    ious = []
    for x, y in dloader:
        x = x.to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).cpu().to(torch.uint8)
        targ = (y > 0.5).to(torch.uint8)
        inter = (pred & targ).sum(dim=(1,2,3)).float()
        union = (pred | targ).sum(dim=(1,2,3)).float()
        iou = (inter + 1e-6) / (union + 1e-6)
        ious.append(iou.mean().item())
    return float(np.mean(ious)) if ious else 0.0

EPOCHS = 3  # tiny, dummy run
for epoch in range(1, EPOCHS+1):
    model.train()
    for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    val_iou = eval_iou(val_dl)
    print(f"Epoch {epoch}: Val IoU = {val_iou:.4f}")

# final test eval
test_iou = eval_iou(test_dl)
print(f"TEST IoU = {test_iou:.4f}")
