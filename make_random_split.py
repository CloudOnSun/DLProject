import os, random

LABEL_DIR = "data/labels"
OUT_DIR   = "."
SEED      = 42
SPLITS    = (0.7, 0.15, 0.15)  # train, val, test

random.seed(SEED)

def main():
    json_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".json")])
    stems = [os.path.splitext(f)[0] for f in json_files]
    random.shuffle(stems)

    n = len(stems)
    n_train = int(SPLITS[0] * n)
    n_val   = int(SPLITS[1] * n)
    train_ids = stems[:n_train]
    val_ids   = stems[n_train:n_train+n_val]
    test_ids  = stems[n_train+n_val:]

    for name, ids in [("train_ids.txt", train_ids),
                      ("val_ids.txt",   val_ids),
                      ("test_ids.txt",  test_ids)]:
        with open(os.path.join(OUT_DIR, name), "w") as f:
            f.write("\n".join(ids))

    print(f"Total: {n}  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

main()
