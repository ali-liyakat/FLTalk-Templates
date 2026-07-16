
# -----------------------------------------
# FLTalk Federated Server (Aggregator Side)
# -----------------------------------------
# type: ignore



from server_agent import fetch_weights, send_global, get_algorithm_code,send_heartbeat, send_log
import time 


# ---------------------------
# User Configurations
# ---------------------------
EXPERIMENT_ID = "exp1"
ALGORITHM = "fedavg"       # "fedavg", "fedavgm", "fedadam"
ROUNDS = 3
NODE_ID = "fed_server"

# number of participating clients per round
NUM_CLIENTS = 2


# ---------------------------
# Fetch Algorithm Code from Backend
# ---------------------------
algo_code = get_algorithm_code(ALGORITHM)
exec(algo_code, globals())  # defines aggregate(weights_list)

print(f"\n Starting Federated Server using '{ALGORITHM}' algorithm")
NODE_ID = "fed_server"


# ---------------------------
# Aggregation Loop
# ---------------------------
for r in range(1, ROUNDS + 1):
    print(f"\n Aggregation Round {r} started...")
    send_heartbeat(EXPERIMENT_ID, "fed_server", NODE_ID, "collecting", r)
    send_log(EXPERIMENT_ID, "fed_server", NODE_ID, f"Aggregation Round {r} started.")


    
    # 1) Collect weights until NUM_CLIENTS reached (accumulate across fetch calls)
    collected = []

    while len(collected) < NUM_CLIENTS:
        send_heartbeat(EXPERIMENT_ID, "fed_server", NODE_ID, "waiting_clients", r)

        weights_batch = fetch_weights(EXPERIMENT_ID)

        if weights_batch:
            collected.extend(weights_batch)
            print(f"Collected {len(collected)}/{NUM_CLIENTS} client weights...")
            send_log(EXPERIMENT_ID, "fed_server", NODE_ID,
                    f"Collected {len(collected)}/{NUM_CLIENTS} client weights")
            
            send_heartbeat(EXPERIMENT_ID, "fed_server", NODE_ID, "collecting", r)
        else:
            print("Waiting for client weights...")
            send_log(EXPERIMENT_ID, "fed_server", NODE_ID, "Waiting for client weights...")

        time.sleep(2)   



    # Use collected weights for aggregation
    weights_list = collected
    print(f"Using {len(weights_list)} client weights for aggregation")


    # Perform aggregation
    global_weights = aggregate(weights_list)

    # Send global model to backend
    send_global(global_weights, EXPERIMENT_ID, round_id=r)
    send_log(EXPERIMENT_ID, "fed_server", NODE_ID, f"Global model sent for round {r}.")
    send_heartbeat(EXPERIMENT_ID, "fed_server", NODE_ID, "round_complete", r)

    print(f"Global model sent.")
    print(f" Round {r} aggregation complete!\n")

