# server.py — koordinator FedAvg + penyedia training data untuk 5 node ESP32
from flask import Flask, request, Response
import numpy as np
import threading
import struct

app = Flask(__name__)

# ===== HARUS SAMA PERSIS DENGAN main.ino DI ESP32 =====
CONTEXT_LEN = 8
VOCAB_SIZE = 128
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LEN
NUM_NODES = 5
MAX_SAMPLES_PER_NODE = 300  # harus <= kapasitas buffer di ESP32

PARAM_COUNT = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM) + (HIDDEN_DIM * VOCAB_SIZE + VOCAB_SIZE)

# ===== STATE GLOBAL: WEIGHT =====
lock = threading.Lock()
global_weights = np.random.uniform(-0.1, 0.1, PARAM_COUNT).astype(np.float32)
pending_updates = {}
round_number = 0

# ===== STATE GLOBAL: TRAINING DATA (dimuat sekali saat startup) =====
node_data = {}  # node_id -> (context_ids: np.uint8 [N, CONTEXT_LEN], target_ids: np.uint16 [N])


def clean_text(text: str) -> str:
    return "".join(c if ord(c) < VOCAB_SIZE else " " for c in text)


def split_corpus(text: str, num_parts: int):
    n = len(text)
    chunk_size = n // num_parts
    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = n if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(text[start:end])
    return chunks


def make_samples(text: str, context_len: int, max_samples: int):
    contexts = []
    targets = []
    for i in range(len(text) - context_len):
        context = text[i:i + context_len]
        target = text[i + context_len]
        contexts.append([ord(c) for c in context])
        targets.append(ord(target))
        if len(contexts) >= max_samples:
            break
    return np.array(contexts, dtype=np.uint8), np.array(targets, dtype=np.uint16)


def load_and_split_corpus(path="corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    text = clean_text(raw)
    chunks = split_corpus(text, NUM_NODES)

    for node_id, chunk in enumerate(chunks, start=1):
        ctx, tgt = make_samples(chunk, CONTEXT_LEN, MAX_SAMPLES_PER_NODE)
        node_data[node_id] = (ctx, tgt)
        print(f"[server] Node {node_id}: {len(tgt)} sample siap dikirim")


@app.route("/get_training_data", methods=["GET"])
def get_training_data():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or node_id not in node_data:
        return Response("node_id tidak valid", status=400)

    ctx, tgt = node_data[node_id]
    num_samples = len(tgt)

    # Format: [uint32 num_samples][context_ids uint8 flat][target_ids uint16]
    payload = struct.pack("<I", num_samples)
    payload += ctx.tobytes()          # num_samples * CONTEXT_LEN byte
    payload += tgt.tobytes()          # num_samples * 2 byte

    return Response(payload, mimetype="application/octet-stream")


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
    expected_bytes = PARAM_COUNT * 4
    if len(raw) != expected_bytes:
        print(f"[server] node {node_id}: ukuran payload salah, dapat {len(raw)}, harusnya {expected_bytes}")
        return Response("ukuran payload tidak cocok", status=400)

    weights = np.frombuffer(raw, dtype=np.float32).copy()

    with lock:
        pending_updates[node_id] = weights
        print(f"[server] Terima weight dari node {node_id} ({len(pending_updates)}/{NUM_NODES} ronde ini)")

        if len(pending_updates) >= NUM_NODES:
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
            "nodes_with_data": list(node_data.keys()),
        }
    return info


if __name__ == "__main__":
    print(f"[server] PARAM_COUNT = {PARAM_COUNT}")
    load_and_split_corpus("corpus.txt")
    print(f"[server] Menunggu {NUM_NODES} node di /upload_weights")
    app.run(host="0.0.0.0", port=5000, threaded=True)