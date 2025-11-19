# make_overfit_subset_tiny.py
import random

SRC = "ids/train_ids.txt"
DST = "ids/overfit_ids.txt"
N = 8 # really tiny, to make sure we CAN memorize

with open(SRC) as f:
    ids = [l.strip() for l in f if l.strip()]

random.seed(0)
N = min(N, len(ids))
subset = random.sample(ids, N)

with open(DST, "w") as f:
    f.write("\n".join(subset))

print(f"Created {DST} with {len(subset)} samples:")
for s in subset:
    print(" ", s)