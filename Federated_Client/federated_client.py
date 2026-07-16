# -----------------------------------------
# # FLTalk Federated Client (Local Training + Evaluation)
# # -----------------------------------------


# type: ignore

from client_agent import push_weights, fetch_global, get_model_code,send_heartbeat, send_log, send_metrics
import time
from data_loader import load_csv_xy, load_mnist, mnist_to_mlp, mnist_to_cnn,load_cifar10_resnet


# ---------------------------
# User Configurations
# ---------------------------
CLIENT_ID = "client_1"
EXPERIMENT_ID = "exp1"
ROUNDS = 3

MODEL_NAME = "mlp_mnist"          ###---  logistic_regression, mlp_2layer, mlp_mnist, cnn_mnist, resnet18  --##

# For CSV-based models (LR / MLP on CSV)
DATA_PATH = "./data/client1_train.csv"

# Hyperparameters (used for torch models; LR may ignore)
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 16

# Dataset split
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------
# Load Local Dataset
# ---------------------------
if MODEL_NAME == "logistic_regression":
    # CSV (tabular)
    X_train, X_test, y_train, y_test = load_csv_xy(
        DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

elif MODEL_NAME == "mlp_2layer":

    # (A) MLP on CSV (tabular)
    X_train, X_test, y_train, y_test = load_csv_xy(
        DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # (B) If you want MLP on MNIST
elif MODEL_NAME == "mlp_mnist":
    X_train, X_test, y_train, y_test = load_mnist(
        data_dir="./data", test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = mnist_to_mlp(X_train)
    X_test  = mnist_to_mlp(X_test)

elif MODEL_NAME == "cnn_mnist":
    # MNIST images
    X_train, X_test, y_train, y_test = load_mnist(
        data_dir="./data", test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = mnist_to_cnn(X_train)
    X_test = mnist_to_cnn(X_test)


elif MODEL_NAME == "resnet18":
    X_train, y_train, X_test, y_test = load_cifar10_resnet(
        data_dir="./data",
        resize_224=True,
        limit_train=200,
        limit_test=100 
    )

else:
    raise ValueError(f"Unsupported MODEL_NAME for current loader: {MODEL_NAME}")


# ---------------------------
# Fetch Model Code from Backend
# ---------------------------
model_code = get_model_code(MODEL_NAME)
exec(model_code, globals())  # defines train_local_model(), update_model(), evaluate_model()

print(f"\nStarting Federated Client [{CLIENT_ID}] using model '{MODEL_NAME}'")


# ---------------------------
# Helper: Safe train call 
# ---------------------------
def _train_call():
    """
    Tries to call train_local_model with hyperparameters.
    If the model function doesn't accept them (e.g., old sklearn LR template),
    it will fall back to calling train_local_model(X_train, y_train) only.
    """
    try:
        return train_local_model(
            X_train,
            y_train,
            epochs=LOCAL_EPOCHS,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
        )
    except TypeError:
        # model function doesn't accept epochs/lr/batch_size
        return train_local_model(X_train, y_train)


# ---------------------------
# Federated Training Loop
# ---------------------------
for r in range(1, ROUNDS + 1):
    print(f"\nRound {r} started...")

    send_heartbeat(EXPERIMENT_ID, "client", CLIENT_ID, "training", r)
    send_log(EXPERIMENT_ID, "client", CLIENT_ID, f"Round {r} started.")

    # 1) Local training
    try:
        local_weights = _train_call()
    except Exception as e:
        print(f"Local training failed: {e}")
        break

    # 2) Push local weights
    resp = push_weights(local_weights, CLIENT_ID, EXPERIMENT_ID)
    if not resp or resp.get("status") != "received":
        print("push_weights failed → round cannot continue")
        time.sleep(2)
        continue

    print(f"Sent local weights: {resp}")
    send_log(EXPERIMENT_ID, "client", CLIENT_ID, f"Sent local weights (round={r}).")
    send_heartbeat(EXPERIMENT_ID, "client", CLIENT_ID, "waiting_global", r)


    # 3) Fetch global weights
    global_weights = fetch_global(EXPERIMENT_ID, expected_round=r)
    if not global_weights:
        print("No global model received → waiting next retry")
        time.sleep(2)
        continue


    send_log(EXPERIMENT_ID, "client", CLIENT_ID, f"Received global model for round {r}.")

    # 4) Update local model with global
    try:
        if "update_model" in globals():
            update_model(global_weights)
    except Exception as e:
        print(f"update_model failed: {e}")



    # 5) Evaluate global model locally (send accuracy + loss)
    acc, loss = None, None

    try:
        out = evaluate_model(global_weights, X_test, y_test)
    except TypeError:
        out = evaluate_model(X_test, y_test)

    # normalize outputs (support: float OR (acc, loss) OR dict)
    if isinstance(out, (tuple, list)):
        if len(out) >= 1: acc = out[0]
        if len(out) >= 2: loss = out[1]
    elif isinstance(out, dict):
        acc = out.get("accuracy", None)
        loss = out.get("loss", None)
    else:
        acc = out

    if acc is not None:
        print(f"[Client {CLIENT_ID}] Eval Accuracy → {float(acc):.4f}")
    if loss is not None:
        print(f"[Client {CLIENT_ID}] Eval Loss     → {float(loss):.4f}")

    payload = {}
    if acc is not None: payload["accuracy"] = float(acc)
    if loss is not None: payload["loss"] = float(loss)

    if payload:
        send_metrics(EXPERIMENT_ID, CLIENT_ID, r, payload)

    send_log(EXPERIMENT_ID, "client", CLIENT_ID, f"Eval done for round {r}.")
    send_heartbeat(EXPERIMENT_ID, "client", CLIENT_ID, "round_complete", r)


    print(f" Round {r} complete!\n")
    time.sleep(2)

print("Client finished.")

