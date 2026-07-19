# server.py — koordinator FedAvg untuk 5 node ESP32
from flask import Flask, request, Response
import numpy as np
import threading

app = Flask(__name__)

# ===== HARUS SAMA PERSIS DENGAN main.ino DI ESP32 =====
CONTEXT_LEN = 8
VOCAB_SIZE = 128
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LEN

# layer1: weights (INPUT_DIM x HIDDEN_DIM) + bias (HIDDEN_DIM)
# layer2: weights (HIDDEN_DIM x VOCAB_SIZE) + bias (VOCAB_SIZE)
PARAM_COUNT = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM) + (HIDDEN_DIM * VOCAB_SIZE + VOCAB_SIZE)

NUM_NODES = 5

# ===== STATE GLOBAL =====
lock = threading.Lock()
global_weights = np.random.uniform(-0.1, 0.1, PARAM_COUNT).astype(np.float32)
pending_updates = {}  # node_id -> np.array weight dari node itu
round_number = 0


@app.route("/get_global_weights", methods=["GET"])
def get_global_weights():
    with lock:
        payload = global_weights.tobytes()
    return Response(payload, mimetype="application/octet-stream")


@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global global_weights, pending_updates, round_number

    node_id = request.args.get("node_id", type=int)
    if node_id is None or not (1 <= node_id <= NUM_NODES):
        return Response("node_id tidak valid", status=400)

    raw = request.get_data()
    expected_bytes = PARAM_COUNT * 4  # float32 = 4 byte
    if len(raw) != expected_bytes:
        print(f"[server] node {node_id}: ukuran payload salah, dapat {len(raw)}, harusnya {expected_bytes}")
        return Response("ukuran payload tidak cocok", status=400)

    weights = np.frombuffer(raw, dtype=np.float32).copy()

    with lock:
        pending_updates[node_id] = weights
        print(f"[server] Terima weight dari node {node_id} ({len(pending_updates)}/{NUM_NODES} ronde ini)")

        if len(pending_updates) >= NUM_NODES:
            # FedAvg: rata-rata sederhana antar node (semua node dianggap bobot sama)
            stacked = np.stack(list(pending_updates.values()), axis=0)
            global_weights = np.mean(stacked, axis=0).astype(np.float32)
            pending_updates.clear()
            round_number += 1
            print(f"[server] === FedAvg selesai, ronde global #{round_number} ===")

    return Response("OK", status=200)


@app.route("/status", methods=["GET"])
def status():
    with lock:
        info = {
            "round_number": round_number,
            "nodes_reported_this_round": list(pending_updates.keys()),
            "param_count": PARAM_COUNT,
        }
    return info


if __name__ == "__main__":
    print(f"[server] PARAM_COUNT = {PARAM_COUNT}")
    print(f"[server] Menunggu {NUM_NODES} node di /upload_weights")
    app.run(host="0.0.0.0", port=5000, threaded=True)