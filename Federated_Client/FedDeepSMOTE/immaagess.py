"""immaagess.py
Generates a visual comparison of representative MNIST samples from three different datasets:"""


import os
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASETS = {
    "Imbalanced": (
        "MNIST/trn_img/0_trn_img.txt",
        "MNIST/trn_lab/0_trn_lab.txt",
    ),

    "Local DeepSMOTE": (
        "MNIST/trn_img_local/0_trn_img.txt",
        "MNIST/trn_lab_local/0_trn_lab.txt",
    ),

    "FedDeepSMOTE": (
        "MNIST/trn_img_f/0_trn_img.txt",
        "MNIST/trn_lab_f/0_trn_lab.txt",
    ),
}

OUTPUT_DIR = "results_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load one representative image per digit
# --------------------------------------------------

def representative_images(img_path, lab_path):

    X = np.loadtxt(img_path)
    y = np.loadtxt(lab_path).astype(int)

    reps = {}

    for digit in range(10):

        idx = np.where(y == digit)[0][0]
        reps[digit] = X[idx].reshape(28,28)

    return reps


# --------------------------------------------------
# Load all datasets
# --------------------------------------------------

all_imgs = {}

for name, (img, lab) in DATASETS.items():
    all_imgs[name] = representative_images(img, lab)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=10,
    figsize=(16, 5)
)

row_names = [
    "Imbalanced",
    "Local\nDeepSMOTE",
    "FedDeepSMOTE"
]

# ---------- Column titles ----------
for c in range(10):
    axes[0, c].set_title(
        str(c),
        fontsize=14,
        fontweight="bold",
        pad=10
    )

# ---------- Images ----------
for r, dataset in enumerate(DATASETS.keys()):

    for c in range(10):

        ax = axes[r, c]

        ax.imshow(
            all_imgs[dataset][c],
            cmap="gray",
            vmin=-1,
            vmax=1
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # Remove borders
        for spine in ax.spines.values():
            spine.set_visible(False)

# ---------- Row labels ----------
for r, label in enumerate(row_names):

    fig.text(
        0.045,                # move left/right
        0.79 - r*0.31,        # vertical position
        label,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

plt.suptitle(
    "Visual Comparison of MNIST Samples",
    fontsize=18,
    fontweight="bold"
)

plt.subplots_adjust(
    left=0.12,
    right=0.99,
    top=0.88,
    bottom=0.05,
    wspace=0.15,
    hspace=0.18
)

outfile = os.path.join(
    OUTPUT_DIR,
    "comparison_digits1.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", outfile)