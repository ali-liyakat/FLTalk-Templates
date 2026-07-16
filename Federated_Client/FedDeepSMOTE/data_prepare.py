""" prepare_data_1fold.py
 Prepare MNIST data in EXACT DeepSMOTE paper/repo format (1-fold only)
 Output:
  MNIST/trn_img/0_trn_img.txt
   MNIST/trn_lab/0_trn_lab.txt
 Run this file FROM INSIDE FedDeepSMOTE folder of each client.
"""

import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from torchvision import datasets, transforms

# -----------------------
# CONFIG (edit if needed)
# -----------------------
SEED = 42

# Make data imbalanced (recommended for DeepSMOTE)
MAKE_IMBALANCED = True

# Same imbalance pattern you already validated
TARGET_COUNTS = {
    0: 4000, 1: 2000, 2: 1000, 3: 750, 4: 500,
    5: 350, 6: 200, 7: 100, 8: 60, 9: 40
}

# Where torchvision will download MNIST
TORCHVISION_ROOT = "./mnist_raw"

# -----------------------
# Helpers
# -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_minus1_plus1(x01):
    # [0,1] -> [-1,1] (DeepSMOTE tanh-compatible)
    return x01 * 2.0 - 1.0


def make_imbalanced(X, y, target_counts):
    per_class_idx = defaultdict(list)
    for i, lab in enumerate(y):
        per_class_idx[int(lab)].append(i)

    selected_idx = []
    for cls, cnt in target_counts.items():
        idxs = per_class_idx.get(cls, [])
        if len(idxs) == 0:
            continue
        if cnt >= len(idxs):
            selected_idx.extend(idxs)
        else:
            selected_idx.extend(random.sample(idxs, cnt))

    random.shuffle(selected_idx)
    return X[selected_idx], y[selected_idx]


# -----------------------
# Main
# -----------------------
def main():
    set_seed(SEED)

    print("Running DeepSMOTE 1-fold data preparation...")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root=TORCHVISION_ROOT,
        train=True,
        download=True,
        transform=transform
    )

    # Load full MNIST training data
    X_list, y_list = [], []
    for img, lab in train_ds:
        X_list.append(img.numpy())   # [1,28,28] in [0,1]
        y_list.append(int(lab))

    X = np.stack(X_list, axis=0)     # (60000,1,28,28)
    y = np.array(y_list)

    print("Original distribution:", dict(Counter(y)))

    # Make imbalanced if required
    if MAKE_IMBALANCED:
        X, y = make_imbalanced(X, y, TARGET_COUNTS)

    print("Imbalanced distribution:", dict(Counter(y)))
    print("Final train size:", X.shape[0])

    # Convert scale for DeepSMOTE
    X = to_minus1_plus1(X)
    X = X.reshape(X.shape[0], -1)    # (N, 784)

    # Prepare DeepSMOTE folder structure
    base_dir = os.path.join(os.getcwd(), "MNIST")
    trn_img_dir = os.path.join(base_dir, "trn_img")
    trn_lab_dir = os.path.join(base_dir, "trn_lab")

    ensure_dir(trn_img_dir)
    ensure_dir(trn_lab_dir)

    img_path = os.path.join(trn_img_dir, "0_trn_img.txt")
    lab_path = os.path.join(trn_lab_dir, "0_trn_lab.txt")

    np.savetxt(img_path, X, fmt="%.6f")
    np.savetxt(lab_path, y.astype(int), fmt="%d")

    print("Saved:")
    print(" ", img_path)
    print(" ", lab_path)
    print("DONE.")


if __name__ == "__main__":
    main()
