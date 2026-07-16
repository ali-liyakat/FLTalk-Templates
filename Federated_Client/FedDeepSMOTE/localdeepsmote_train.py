"""local_deepsmote_train.py
Solution-1 (Local DeepSMOTE): TRAIN encoder/decoder locally on this client's data.

Run inside: Client_i/FedDeepSMOTE/
  python local_deepsmote_train.py

Input:
  ./MNIST/trn_img/0_trn_img.txt
  ./MNIST/trn_lab/0_trn_lab.txt

Output (LOCAL checkpoints):
  ./checkpoints_local/<CLIENT_ID>/local_enc_final.pth
  ./checkpoints_local/<CLIENT_ID>/local_dec_final.pth
  """



import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================
# CONFIG (edit per client)
# =========================
CLIENT_ID = "client_1"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5              
LR = 2e-4
BATCH_SIZE = 100

# DeepSMOTE txt (1-fold) — already in your FedDeepSMOTE folder
TRN_IMG_FILE = "./MNIST/trn_img/0_trn_img.txt"
TRN_LAB_FILE = "./MNIST/trn_lab/0_trn_lab.txt"

# Save LOCAL checkpoints here (won't clash with FedDeepSMOTE)
OUT_DIR = f"./Local_Models"
SAVE_EACH_EPOCH = False  # if False -> only saves final

# latent dim must match your FedDeepSMOTE pipeline
DIM_H = 64
N_CHANNEL = 1
N_Z = 300


# =========================
# Models (must match your FedDeepSMOTE training)
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
        # MNIST 28x28 -> after 4 downsamples -> 1x1 spatial, channels dim_h*8
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
            nn.Tanh(),  # DeepSMOTE expects [-1,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        return self.deconv(x)


def load_local_txt():
    if not os.path.isfile(TRN_IMG_FILE) or not os.path.isfile(TRN_LAB_FILE):
        raise FileNotFoundError(
            f"Missing data files:\n{TRN_IMG_FILE}\n{TRN_LAB_FILE}\n"
            "Run data_prepare.py first inside this FedDeepSMOTE folder."
        )

    X = np.loadtxt(TRN_IMG_FILE).astype(np.float32)   # (N,784) in [-1,1]
    y = np.loadtxt(TRN_LAB_FILE).astype(np.int64)     # (N,)
    X = X.reshape(-1, 1, 28, 28)
    return X, y


def train_local_deepsmote(encoder, decoder, X, y):
    encoder.train()
    decoder.train()

    criterion = nn.MSELoss().to(DEVICE)
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=LR)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=LR)

    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(tensor_x, tensor_y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # used for mse2 (neighbor reconstruction term)
    dec_x = X
    dec_y = y

    for ep in range(EPOCHS):
        total = total_mse = total_mse2 = 0.0

        for images, _ in dl:
            images = images.to(DEVICE)

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # mse reconstruction
            z_hat = encoder(images)
            x_hat = decoder(z_hat)
            mse = criterion(x_hat, images)

            # mse2 "neighbor reconstruction" (DeepSMOTE style)
            tc = int(np.random.choice(10, 1)[0])
            xbeg = dec_x[dec_y == tc]
            xlen = len(xbeg)

            if xlen < 2:
                mse2 = torch.tensor(0.0, device=DEVICE)
            else:
                nsamp = min(xlen, 100)
                ind = np.random.choice(np.arange(xlen), nsamp, replace=False)
                xclass = xbeg[ind]

                xclen = len(xclass)
                xcminus = np.arange(1, xclen)
                xcplus = np.append(xcminus, 0)

                xcnew = xclass[[*xcplus]].copy()
                xcnew_t = torch.tensor(xcnew, dtype=torch.float32).to(DEVICE)

                xclass_t = torch.tensor(xclass, dtype=torch.float32).to(DEVICE)
                xclass_enc = encoder(xclass_t).detach().cpu().numpy()

                xc_enc = xclass_enc[[*xcplus]].copy()
                xc_enc_t = torch.tensor(xc_enc, dtype=torch.float32).to(DEVICE)

                ximg = decoder(xc_enc_t)
                mse2 = criterion(ximg, xcnew_t)

            loss = mse + mse2
            loss.backward()
            enc_optim.step()
            dec_optim.step()

            total += float(loss.item()) * images.size(0)
            total_mse += float(mse.item()) * images.size(0)
            total_mse2 += float(mse2.item()) * images.size(0)

        avg = total / len(dl)
        avg_mse = total_mse / len(dl)
        avg_mse2 = total_mse2 / len(dl)

        print(f"Epoch {ep}: loss={avg:.6f} mse={avg_mse:.6f} mse2={avg_mse2:.6f}")

        if SAVE_EACH_EPOCH:
            os.makedirs(OUT_DIR, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                       os.path.join(OUT_DIR, f"local_enc_epoch_{ep}.pth"))
            torch.save({k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                       os.path.join(OUT_DIR, f"local_dec_epoch_{ep}.pth"))
            print(f"Saved epoch checkpoints to {OUT_DIR}")

    return encoder, decoder


def main():
    print(f"[Local DeepSMOTE TRAIN] CLIENT_ID={CLIENT_ID} device={DEVICE}")
    X, y = load_local_txt()
    print("Data:", X.shape, y.shape)

    encoder = Encoder(dim_h=DIM_H, n_channel=N_CHANNEL, n_z=N_Z).to(DEVICE)
    decoder = Decoder(dim_h=DIM_H, n_channel=N_CHANNEL, n_z=N_Z).to(DEVICE)

    t0 = time.time()
    encoder, decoder = train_local_deepsmote(encoder, decoder, X, y)
    print(f"Training finished in {(time.time()-t0)/60:.2f} min")

    # Save final checkpoints (always)
    os.makedirs(OUT_DIR, exist_ok=True)
    enc_path = os.path.join(OUT_DIR, "local_enc_final.pth")
    dec_path = os.path.join(OUT_DIR, "local_dec_final.pth")

    torch.save({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, enc_path)
    torch.save({k: v.detach().cpu() for k, v in decoder.state_dict().items()}, dec_path)

    print("Saved FINAL local checkpoints:")
    print(" ", os.path.abspath(enc_path))
    print(" ", os.path.abspath(dec_path))


if __name__ == "__main__":
    main()
