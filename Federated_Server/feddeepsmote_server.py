"""feddeepsmote_server.py
FedDeepSMOTE Aggregation Server (AUTO ROUNDS)

Run from FLTalk_Server folder:
  python feddeepsmote_server.py

It will:
  1) send initial global for round=1
  2) for r in 1..ROUNDS:
       wait until NUM_CLIENTS updates for round r arrive
       FedAvg enc/dec
       send global for round r+1
       """



import io
import time
import base64
from typing import Dict, Any, List

import torch
import torch.nn as nn

from server_agent import fetch_feddeepsmote_weights, send_feddeepsmote_global


# =========================
# CONFIG
# =========================
EXPERIMENT_ID = "default"
NUM_CLIENTS = 2
ROUNDS = 5

# seconds between polling backend for updates
POLL_INTERVAL_SEC = 2


# --------------------------
# Models (must match client-side exactly)
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
# Serialization helpers: torch state_dict <-> base64
# --------------------------
def state_dict_to_b64(sd: Dict[str, torch.Tensor]) -> str:
    bio = io.BytesIO()
    torch.save(sd, bio)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def b64_to_state_dict(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("utf-8"))
    bio = io.BytesIO(raw)
    return torch.load(bio, map_location="cpu")


# --------------------------
# FedAvg for state_dict
# --------------------------
def fedavg_state_dict(weighted_sds: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not weighted_sds:
        raise ValueError("No client state_dicts provided for FedAvg.")

    total_n = sum(int(x["n"]) for x in weighted_sds)
    if total_n <= 0:
        raise ValueError("Total n_samples must be > 0.")

    keys = weighted_sds[0]["sd"].keys()
    out: Dict[str, torch.Tensor] = {}

    for k in keys:
        t0 = weighted_sds[0]["sd"][k]
        if not torch.is_tensor(t0):
            out[k] = t0
            continue

        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0, dtype=torch.float32)
            for item in weighted_sds:
                sd = item["sd"]
                w = float(item["n"]) / float(total_n)
                acc += sd[k].detach().cpu().to(torch.float32) * w
            out[k] = acc.to(dtype=t0.dtype)
        else:
            out[k] = t0.detach().cpu()

    return out


def collect_round_updates(round_id: int) -> Dict[str, Dict[str, Any]]:
    """
    Keep polling backend until we have NUM_CLIENTS unique client updates for round_id.
    Returns mapping: client_id -> update_dict
    """
    per_client: Dict[str, Dict[str, Any]] = {}

    while len(per_client) < NUM_CLIENTS:
        # backend returns either waiting or ready; the agent already waits,
        # but it returns immediately when ready, so we still guard here.
        updates = fetch_feddeepsmote_weights(experiment_id=EXPERIMENT_ID)

        for u in updates:
            cid = u.get("client_id", "unknown")
            ur = int(u.get("round", round_id))
            if ur != round_id:
                continue
            per_client[cid] = u

        print(f"[FedDeepSMOTE] Round {round_id}: collected {len(per_client)}/{NUM_CLIENTS}")
        if len(per_client) < NUM_CLIENTS:
            time.sleep(POLL_INTERVAL_SEC)

    return per_client


def run():
    # 1) Initialize global (round=1)
    enc = Encoder()
    dec = Decoder()

    global_payload = {
        "enc": state_dict_to_b64(enc.state_dict()),
        "dec": state_dict_to_b64(dec.state_dict()),
    }

    print("[FedDeepSMOTE] Sending INITIAL global (round=1)")
    send_feddeepsmote_global(global_payload, experiment_id=EXPERIMENT_ID, round_id=1)

    # 2) Rounds loop
    for r in range(1, ROUNDS + 1):
        print(f"\n[FedDeepSMOTE] ===== Round {r}/{ROUNDS} =====")
        print(f"[FedDeepSMOTE] Waiting for {NUM_CLIENTS} client updates...")

        per_client = collect_round_updates(r)

        enc_list = []
        dec_list = []

        for cid, u in per_client.items():
            n = int(u.get("n_samples", 0))
            if n <= 0:
                raise ValueError(f"Client {cid} sent invalid n_samples={n}")

            enc_sd = b64_to_state_dict(u["enc"])
            dec_sd = b64_to_state_dict(u["dec"])

            enc_list.append({"sd": enc_sd, "n": n})
            dec_list.append({"sd": dec_sd, "n": n})

        new_enc = fedavg_state_dict(enc_list)
        new_dec = fedavg_state_dict(dec_list)

        next_round = r + 1
        new_payload = {
            "enc": state_dict_to_b64(new_enc),
            "dec": state_dict_to_b64(new_dec),
        }

        # print(f"[FedDeepSMOTE] Aggregated. Sending global (round={next_round})")
        send_feddeepsmote_global(new_payload, experiment_id=EXPERIMENT_ID, round_id=next_round)

    print("\n[FedDeepSMOTE] Done.")


if __name__ == "__main__":
    run()
