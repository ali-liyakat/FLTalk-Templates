""" -----------------------------------------
 FLTalk Server Agent
 Communication module between FLTalk Server and FLTalk Main server.
 -----------------------------------------"""


import requests
import time

# ---------------------------------------------------
# MAIN SERVER URL
# ---------------------------------------------------
MAIN_SERVER = "http://127.0.0.1:8000" 
# MAIN_SERVER = "https://fltalk-system.onrender.com"


# ---------------------------------------------------
# FETCH CLIENT WEIGHTS FOR AGGREGATION
# ---------------------------------------------------
def fetch_weights(experiment_id="default", retry_interval=2):
    while True:
        try:
            res = requests.get(
                f"{MAIN_SERVER}/fetch_weights",
                params={"experiment_id": experiment_id},
                timeout=60
            )
            data = res.json()

            # When server has weights
            if data.get("status") == "ready":
                weights = data["weights"]
                print(f"Received {len(weights)} client weights")
                return weights

            # print("Waiting for client weights...")
            time.sleep(retry_interval)

        except Exception as e:
            print("fetch_weights error:", e)
            time.sleep(retry_interval)




def send_global(global_weights, experiment_id="default", round_id=1):
    try:
        res = requests.post(
            f"{MAIN_SERVER}/send_global",
            json={"experiment_id": experiment_id, "global_weights": global_weights, "round": round_id},
            timeout=60
        )
        out = res.json()
        print("Global weights sent:", out)
        return out

    except Exception as e:
        print("send_global error:", e)
        return None




# =========================================================
# FedDeepSMOTE Server APIs (Skeleton)
# =========================================================

def fetch_feddeepsmote_weights(experiment_id="default", retry_interval=2):
    while True:
        try:
            res = requests.get(
                f"{MAIN_SERVER}/fetch_feddeepsmote_weights",
                params={"experiment_id": experiment_id},
                timeout=60
            )
            data = res.json()

            if data.get("status") == "ready":
                updates = data["updates"]
                print(f"Received {len(updates)} FedDeepSMOTE updates")
                return updates

            # print("Waiting for FedDeepSMOTE updates...")
            time.sleep(retry_interval)

        except Exception as e:
            print("fetch_feddeepsmote_weights error:", e)
            time.sleep(retry_interval)


def send_feddeepsmote_global(global_payload, experiment_id="default", round_id=1):
    """
    global_payload expected: {"enc": ..., "dec": ...}
    """
    try:
        res = requests.post(
            f"{MAIN_SERVER}/send_feddeepsmote_global",
            json={"experiment_id": experiment_id, "global": global_payload, "round": round_id},
            timeout=60
        )
        out = res.json()
        print("FedDeepSMOTE global sent:", out)
        return out

    except Exception as e:
        print("send_feddeepsmote_global error:", e)
        return None



# ---------------------------------------------------
# FETCH AGGREGATION ALGORITHM CODE (IMPORTANT!)
# ---------------------------------------------------
def get_algorithm_code(algo_name):
    try:
        res = requests.get(f"{MAIN_SERVER}/get_algorithm_code", params={"algo_name": algo_name})
        data = res.json()
        if data.get("status") == "ok":
            print(f"Algorithm '{algo_name}' code fetched successfully")
            return data["code"]
        print("Algorithm fetch failed:", data)
    except Exception as e:
        print("get_algorithm_code failed:", e)


def send_heartbeat(experiment_id, node_type, node_id, status, round_id=0):
    # print(f"[UI_AGENT] Heartbeat -> {node_type}:{node_id} round={round_id}")

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
            timeout=2,
        )
    except Exception as e:
        print("[UI_AGENT] Heartbeat failed:", e)


def send_log(experiment_id, node_type, node_id, message):
    # print(f"[UI_AGENT] Log -> {node_type}:{node_id} :: {message}")

    try:
        requests.post(
            f"{MAIN_SERVER}/ui/log",
            json={
                "experiment_id": experiment_id,
                "node_type": node_type,
                "node_id": node_id,
                "message": message,
            },
            timeout=2,
        )
    except Exception as e:
        print("[UI_AGENT] Log failed:", e)