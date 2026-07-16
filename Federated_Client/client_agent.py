""" -----------------------------------------
 FLTalk Client Agent
 Communication module between Fedearted_Client and FLTalk Main server.
 -----------------------------------------
"""

import requests
import json
import time

# Backend server URL
MAIN_SERVER = "http://127.0.0.1:8000"   
# MAIN_SERVER = "https://fltalk-system.onrender.com"


# --------------------------------------------------
# SEND LOCAL CLIENT WEIGHTS TO MAIN SERVER
# --------------------------------------------------

def push_weights(weights, client_id, experiment_id="default"):
    try:
        res = requests.post(
            f"{MAIN_SERVER}/send_weights",
            json={"client_id": client_id, "weights": weights, "experiment_id": experiment_id},
            timeout=60
        )
        data = res.json()
        # print("ent local weights:", data)
        return data
    except Exception as e:
        print("push_weights failed:", e)
        return None



def fetch_global(experiment_id="default", expected_round=1, retry_interval=2):
    while True:
        try:
            res = requests.get(
                f"{MAIN_SERVER}/fetch_global",
                params={"experiment_id": experiment_id, "expected_round": expected_round},
                timeout=60
            )
            data = res.json()

            if data.get("status") == "ready":
                got_round = int(data.get("round", 0))
                if got_round >= int(expected_round):
                    print(f"Global model received! (round={got_round})")
                    return data["global_weights"]

            # still waiting for correct round
            current_round = data.get("current_round", 0)
            # print(f"Waiting for global model... (expected={expected_round}, current={current_round})")
            time.sleep(retry_interval)

        except Exception as e:
            print("fetch_global error:", e)
            time.sleep(retry_interval)





# =========================================================
# FedDeepSMOTE Client APIs (Skeleton)
# =========================================================

def push_feddeepsmote_weights(enc, dec, client_id, experiment_id="default", round_id=0, n_samples=0):
    try:
        res = requests.post(
            f"{MAIN_SERVER}/send_feddeepsmote_weights",
            json={
                "client_id": client_id,
                "experiment_id": experiment_id,
                "round": round_id,
                "n_samples": n_samples,
                "enc": enc,
                "dec": dec,
            },
            timeout=60
        )
        data = res.json()
        print("Sent FedDeepSMOTE update:", data)
        return data
    except Exception as e:
        print("push_feddeepsmote_weights failed:", e)
        return None


def fetch_feddeepsmote_global(experiment_id="default", expected_round=1, retry_interval=2):
    while True:
        try:
            res = requests.get(
                f"{MAIN_SERVER}/fetch_feddeepsmote_global",
                params={"experiment_id": experiment_id, "expected_round": expected_round},
                timeout=60
            )
            data = res.json()

            if data.get("status") == "ready":
                got_round = int(data.get("round", 0))
                if got_round >= int(expected_round):
                    print(f"FedDeepSMOTE global received! (round={got_round})")
                    return data["global"]

            current_round = data.get("current_round", 0)
            # print(f"Waiting for FedDeepSMOTE global... (expected={expected_round}, current={current_round})")
            time.sleep(retry_interval)

        except Exception as e:
            print("fetch_feddeepsmote_global error:", e)
            time.sleep(retry_interval)



# ---------------------------
# Get model code from backend
# ---------------------------
def get_model_code(model_name):
    try:
        res = requests.get(f"{MAIN_SERVER}/get_model_code", params={"model_name": model_name})
        data = res.json()
        if data.get("status") == "ok":
            print(f"Model '{model_name}' code fetched successfully")
            return data["code"]
        print("Model fetch failed:", data)
    except Exception as e:
        print("get_model_code failed:", e)



def send_heartbeat(experiment_id: str, node_type: str, node_id: str, status: str, round_id: int = 0):
    try:
        requests.post(
            f"{MAIN_SERVER}/ui/heartbeat",
            json={
                "experiment_id": experiment_id,
                "node_type": node_type,
                "node_id": node_id,
                "status": status,
                "round": round_id,
            },
            timeout=3,
        )
    except Exception:
        pass

def send_log(experiment_id: str, node_type: str, node_id: str, message: str):
    try:
        requests.post(
            f"{MAIN_SERVER}/ui/log",
            json={
                "experiment_id": experiment_id,
                "node_type": node_type,
                "node_id": node_id,
                "message": message,
            },
            timeout=3,
        )
    except Exception:
        pass


def send_metrics(experiment_id: str, node_id: str, round_id: int, metrics: dict):
    try:
        requests.post(
            f"{MAIN_SERVER}/ui/metrics",
            json={
                "experiment_id": experiment_id,
                "node_id": node_id,
                "round": round_id,
                "metrics": metrics,
            },
            timeout=3,
        )
    except Exception:
        pass
