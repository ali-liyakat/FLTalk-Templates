""" feddeepsmote_client.py
 FedDeepSMOTE Client Runner (AUTO ROUNDS)

 Run inside each client's FedDeepSMOTE folder:
   python feddeepsmote_client.py

 It will:
   for r in 1..ROUNDS:
       fetch global (round=r)
       train locally (LOCAL_EPOCHS_PER_ROUND)
       push updated enc/dec (round=r)
"""

import os
import sys
import time
import base64
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================
# CONFIG (edit per client)
# =========================
CLIENT_ID = "client_1"    
EXPERIMENT_ID = "default"
ROUNDS = 5                     # total FedDeepSMOTE rounds to run
LOCAL_EPOCHS_PER_ROUND = 5
LR = 2e-4
BATCH_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How long to wait between polling global weights
GLOBAL_POLL_INTERVAL_SEC = 2
# Save ONLY final global enc/dec on client
SAVE_FINAL_GLOBAL = True
GLOBAL_SAVE_DIR = "Models"


# --------------------------
# Import client_agent.py
# (client_agent.py is in Client_i/ folder)
# --------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from client_agent import push_feddeepsmote_weights, fetch_feddeepsmote_global  # noqa


# --------------------------
# Model (must match server-side)
# --------------------------
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


# --------------------------
# Serialization helpers
# --------------------------
def state_dict_to_b64(sd: dict) -> str:
    import io
    bio = io.BytesIO()
    torch.save(sd, bio)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def b64_to_state_dict(b64: str) -> dict:
    import io
    raw = base64.b64decode(b64.encode("utf-8"))
    bio = io.BytesIO(raw)
    return torch.load(bio, map_location="cpu")


def maybe_decode_payload(x):
    if isinstance(x, str):
        return b64_to_state_dict(x)
    if isinstance(x, dict):
        # (for earlier dummy testing only)
        return x
    raise TypeError(f"Unsupported payload type: {type(x)}")


# --------------------------
# Local data loading (DeepSMOTE txt format)
# --------------------------
def load_local_mnist_txt(base_dir: str):
    img_path = os.path.join(base_dir, "trn_img", "0_trn_img.txt")
    lab_path = os.path.join(base_dir, "trn_lab", "0_trn_lab.txt")
    if not os.path.isfile(img_path) or not os.path.isfile(lab_path):
        raise FileNotFoundError(f"Missing MNIST txt files:\n{img_path}\n{lab_path}")

    X = np.loadtxt(img_path).astype(np.float32)   # (N,784) in [-1,1]
    y = np.loadtxt(lab_path).astype(np.int64)     # (N,)
    X = X.reshape(-1, 1, 28, 28)
    return X, y


# --------------------------
# DeepSMOTE local training loop
# --------------------------
def train_local_deepsmote(encoder, decoder, X, y, device, epochs=1, lr=2e-4, batch_size=100):
    encoder.train()
    decoder.train()

    criterion = nn.MSELoss().to(device)
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr)

    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(tensor_x, tensor_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    dec_x = X
    dec_y = y

    for ep in range(epochs):
        total = total_mse = total_mse2 = 0.0

        for images, _ in dl:
            images = images.to(device)

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            z_hat = encoder(images)
            x_hat = decoder(z_hat)
            mse = criterion(x_hat, images)

            tc = int(np.random.choice(10, 1)[0])
            xbeg = dec_x[dec_y == tc]
            xlen = len(xbeg)

            if xlen < 2:
                mse2 = torch.tensor(0.0, device=device)
            else:
                nsamp = min(xlen, 100)
                ind = np.random.choice(np.arange(xlen), nsamp, replace=False)
                xclass = xbeg[ind]

                xclen = len(xclass)
                xcminus = np.arange(1, xclen)
                xcplus = np.append(xcminus, 0)

                xcnew = xclass[[*xcplus]].copy()
                xcnew_t = torch.tensor(xcnew, dtype=torch.float32).to(device)

                xclass_t = torch.tensor(xclass, dtype=torch.float32).to(device)
                xclass_enc = encoder(xclass_t).detach().cpu().numpy()

                xc_enc = xclass_enc[[*xcplus]].copy()
                xc_enc_t = torch.tensor(xc_enc, dtype=torch.float32).to(device)

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

    return encoder, decoder


def run():
    print(f"[Client {CLIENT_ID}]")
    mnist_dir = os.path.join(THIS_DIR, "MNIST")
    X, y = load_local_mnist_txt(mnist_dir)
    n_samples = int(X.shape[0])
    # print(f"[Client {CLIENT_ID}] Local train data: {X.shape}, n_samples={n_samples}")

    for r in range(1, ROUNDS + 1):
        print(f"\n[Client {CLIENT_ID}] ===== Round {r}/{ROUNDS} =====")

        global_payload = fetch_feddeepsmote_global(
            experiment_id=EXPERIMENT_ID,
            expected_round=r,
            retry_interval=GLOBAL_POLL_INTERVAL_SEC
        )

        if not isinstance(global_payload, dict) or "enc" not in global_payload or "dec" not in global_payload:
            raise ValueError("Global payload must be {'enc':..., 'dec':...}")

        enc_sd = maybe_decode_payload(global_payload["enc"])
        dec_sd = maybe_decode_payload(global_payload["dec"])

        encoder = Encoder().to(DEVICE)
        decoder = Decoder().to(DEVICE)

        encoder.load_state_dict(enc_sd, strict=True)
        decoder.load_state_dict(dec_sd, strict=True)
        # print(f"[Client {CLIENT_ID}] Loaded global enc/dec for round {r}")

        t0 = time.time()
        encoder, decoder = train_local_deepsmote(
            encoder, decoder, X, y,
            device=DEVICE,
            epochs=LOCAL_EPOCHS_PER_ROUND,
            lr=LR,
            batch_size=BATCH_SIZE
        )
        # print(f"[Client {CLIENT_ID}] Local training done in {(time.time()-t0):.2f}s")

        enc_b64 = state_dict_to_b64(encoder.state_dict())
        dec_b64 = state_dict_to_b64(decoder.state_dict())

        resp = push_feddeepsmote_weights(
            enc=enc_b64,
            dec=dec_b64,
            client_id=CLIENT_ID,
            experiment_id=EXPERIMENT_ID,
            round_id=r,
            n_samples=n_samples
        )
        # print(f"[Client {CLIENT_ID}] Push response: {resp}")

    # =========================================================
# Save TRUE FINAL GLOBAL (after last aggregation)
# =========================================================
    if SAVE_FINAL_GLOBAL:
        final_round = ROUNDS + 1  # 4 when ROUNDS=3
        print(f"\n[Client {CLIENT_ID}] Fetching TRUE FINAL GLOBAL (round={final_round})...")

        final_global = fetch_feddeepsmote_global(
            experiment_id=EXPERIMENT_ID,
            expected_round=final_round,
            retry_interval=GLOBAL_POLL_INTERVAL_SEC
        )

        enc_sd_final = maybe_decode_payload(final_global["enc"])
        dec_sd_final = maybe_decode_payload(final_global["dec"])

        save_dir = os.path.join(THIS_DIR, GLOBAL_SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)

        enc_path = os.path.join(save_dir, f"global_enc_final.pth")
        dec_path = os.path.join(save_dir, f"global_dec_final.pth")

        torch.save({k: v.detach().cpu() for k, v in enc_sd_final.items()}, enc_path)
        torch.save({k: v.detach().cpu() for k, v in dec_sd_final.items()}, dec_path)

        print(f"[Client {CLIENT_ID}] Saved TRUE FINAL GLOBAL")

        print(f"\n[Client {CLIENT_ID}] Done.")


if __name__ == "__main__":
    run()
