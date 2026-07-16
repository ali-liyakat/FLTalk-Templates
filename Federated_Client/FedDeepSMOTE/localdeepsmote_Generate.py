"""local_deepsmote_generate.py
Solution-1 (Local DeepSMOTE): GENERATE balanced data using locally-trained encoder/decoder.

Run inside: Client_i/FedDeepSMOTE/
  python local_deepsmote_generate.py

Input (imbalanced DeepSMOTE format):
  ./MNIST/trn_img/0_trn_img.txt
  ./MNIST/trn_lab/0_trn_lab.txt

Local model checkpoints (trained by local_deepsmote_train.py):
  ./Local_Models/local_enc_final.pth
  ./Local_Models/local_dec_final.pth

Output (balanced dataset, LOCAL DeepSMOTE):
  ./MNIST/trn_img_local/0_trn_img.txt
  ./MNIST/trn_lab_local/0_trn_lab.txt
  """



import collections
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

t0 = time.time()
np.printoptions(precision=5, suppress=True)

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data (1-fold)
TRN_IMG_FILE = "./MNIST/trn_img/0_trn_img.txt"
TRN_LAB_FILE = "./MNIST/trn_lab/0_trn_lab.txt"

# Local checkpoints
ENC_PATH = "./Local_Models/local_enc_final.pth"
DEC_PATH = "./Local_Models/local_dec_final.pth"

# Output folders (LOCAL DeepSMOTE)
OUT_IMG_FILE = "./MNIST/trn_img_local/0_trn_img.txt"
OUT_LAB_FILE = "./MNIST/trn_lab_local/0_trn_lab.txt"

# Target class counts used by DeepSMOTE repo (majority=4000)
imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

# Latent dimension must match training
DIM_H = 64
N_CHANNEL = 1
N_Z = 300


# =========================
# Models (must match training architecture)
# =========================
class Encoder(nn.Module):
    def __init__(self, dim_h=64, n_channel=1, n_z=300):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim_h, dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim_h * 2, dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim_h * 4, dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(dim_h * (2 ** 3), n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, dim_h=64, n_channel=1, n_z=300):
        super().__init__()
        self.dim_h = dim_h
        self.fc = nn.Sequential(
            nn.Linear(n_z, dim_h * 8 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim_h * 8, dim_h * 4, 4),
            nn.BatchNorm2d(dim_h * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim_h * 4, dim_h * 2, 4),
            nn.BatchNorm2d(dim_h * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim_h * 2, 1, 4, stride=2),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        return self.deconv(x)


# =========================
# Helpers (latent-space SMOTE)
# =========================
def biased_get_class1(dec_x, dec_y, c):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg


def G_SM1(X, n_to_sample, cl):
    # Nearest neighbor interpolation in latent space
    n_neigh = 6  # 5 + self
    nnm = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nnm.fit(X)
    _, ind = nnm.kneighbors(X)

    base_indices = np.random.choice(np.arange(len(X)), n_to_sample)
    neighbor_indices = np.random.choice(np.arange(1, n_neigh), n_to_sample)

    X_base = X[base_indices]
    X_nei = X[ind[base_indices, neighbor_indices]]

    lam = np.random.rand(n_to_sample, 1)
    samples = X_base + lam * (X_nei - X_base)
    return samples, np.full((n_to_sample,), cl, dtype=np.int64)


def main():
    print("device:", DEVICE)

    # 1) Load local imbalanced dataset
    if not os.path.isfile(TRN_IMG_FILE) or not os.path.isfile(TRN_LAB_FILE):
        raise FileNotFoundError(
            f"Missing data files:\n{TRN_IMG_FILE}\n{TRN_LAB_FILE}\n"
            "Run data_prepare.py first inside this FedDeepSMOTE folder."
        )

    dec_x = np.loadtxt(TRN_IMG_FILE).astype(np.float32)  # (N,784)
    dec_y = np.loadtxt(TRN_LAB_FILE).astype(np.int64)    # (N,)

    print("train imgs before reshape ", dec_x.shape)
    print("train labels ", dec_y.shape)
    print(collections.Counter(dec_y))

    dec_x = dec_x.reshape(dec_x.shape[0], 1, 28, 28)
    print("train imgs after reshape ", dec_x.shape)

    # 2) Load LOCAL trained encoder/decoder
    if not os.path.isfile(ENC_PATH) or not os.path.isfile(DEC_PATH):
        raise FileNotFoundError(
            f"Missing LOCAL model checkpoints:\n{ENC_PATH}\n{DEC_PATH}\n"
            "Run local_deepsmote_train.py first."
        )

    encoder = Encoder(dim_h=DIM_H, n_channel=N_CHANNEL, n_z=N_Z).to(DEVICE)
    decoder = Decoder(dim_h=DIM_H, n_channel=N_CHANNEL, n_z=N_Z).to(DEVICE)

    enc_sd = torch.load(ENC_PATH, map_location="cpu")
    dec_sd = torch.load(DEC_PATH, map_location="cpu")

    encoder.load_state_dict(enc_sd, strict=True)
    decoder.load_state_dict(dec_sd, strict=True)

    encoder.eval()
    decoder.eval()

    # 3) Generate synthetic samples for classes 1..9 to reach majority count
    resx = []
    resy = []

    with torch.no_grad():
        for i in range(1, 10):
            xclass, yclass = biased_get_class1(dec_x, dec_y, i)
            print(xclass.shape)

            if len(yclass) == 0:
                print(f"Class {i} missing locally, skip.")
                continue

            xclass_t = torch.tensor(xclass, dtype=torch.float32, device=DEVICE)
            z = encoder(xclass_t).detach().cpu().numpy()

            n = imbal[0] - imbal[i]
            if n <= 0:
                print(f"Class {i} already >= majority target, skip.")
                continue

            zsamp, ysamp = G_SM1(z, n_to_sample=n, cl=i)

            zsamp_t = torch.tensor(zsamp, dtype=torch.float32, device=DEVICE)
            ximg = decoder(zsamp_t).detach().cpu().numpy()  # (n,1,28,28)

            resx.append(ximg)
            resy.append(ysamp)

    if len(resx) == 0:
        raise RuntimeError("No synthetic samples generated. Check class distribution.")

    resx1 = np.vstack(resx)             # (Nsyn,1,28,28)
    resy1 = np.hstack(resy)             # (Nsyn,)

    resx1 = resx1.reshape(resx1.shape[0], -1)  # (Nsyn,784)
    dec_x1 = dec_x.reshape(dec_x.shape[0], -1) # (Norig,784)

    combx = np.vstack((resx1, dec_x1))
    comby = np.hstack((resy1, dec_y))

    print("Final balanced shapes:", combx.shape, comby.shape)
    print("Final class counts:", collections.Counter(comby))

    # 4) Save outputs
    os.makedirs(os.path.dirname(OUT_IMG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LAB_FILE), exist_ok=True)

    np.savetxt(OUT_IMG_FILE, combx)
    np.savetxt(OUT_LAB_FILE, comby)

    print("Saved:")
    print(" ", os.path.abspath(OUT_IMG_FILE))
    print(" ", os.path.abspath(OUT_LAB_FILE))

    print("final time(min): {:.2f}".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
