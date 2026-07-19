# server.py — koordinator FedAvg + penyedia training data + deteksi training selesai
from flask import Flask, request, Response
import numpy as np
import threading
import struct
import os
import math

app = Flask(__name__)

# ===== HARUS SAMA PERSIS DENGAN main.ino DI ESP32 =====
CONTEXT_LEN = 8
VOCAB_SIZE = 128
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LEN
NUM_NODES = 5
MAX_SAMPLES_PER_NODE = 300

PARAM_COUNT = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM) + (HIDDEN_DIM * VOCAB_SIZE + VOCAB_SIZE)
W1_SIZE = INPUT_DIM * HIDDEN_DIM
B1_SIZE = HIDDEN_DIM
W2_SIZE = HIDDEN_DIM * VOCAB_SIZE
B2_SIZE = VOCAB_SIZE

CHECKPOINT_DIR = "checkpoints"

# ===== KRITERIA BERHENTI TRAINING =====
TARGET_LOSS = 3.0
MAX_ROUNDS = 50
EVAL_HOLDOUT_FRACTION = 0.15

# Baseline loss teoretis kalau model tebak acak (uniform) dari VOCAB_SIZE kelas —
# dipakai sebagai titik awal 0% buat progress bar loss
BASELINE_LOSS = math.log(VOCAB_SIZE)

# ===== STATE GLOBAL: WEIGHT =====
lock = threading.Lock()
global_weights = np.random.uniform(-0.1, 0.1, PARAM_COUNT).astype(np.float32)
pending_updates = {}
round_number = 0
training_complete = False
last_eval_loss = None

# ===== STATE GLOBAL: DATA =====
node_data = {}
eval_context = None
eval_target = None


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
    global eval_context, eval_target

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    text = clean_text(raw)

    split_point = int(len(text) * (1.0 - EVAL_HOLDOUT_FRACTION))
    train_text = text[:split_point]
    eval_text = text[split_point:]

    chunks = split_corpus(train_text, NUM_NODES)
    for node_id, chunk in enumerate(chunks, start=1):
        ctx, tgt = make_samples(chunk, CONTEXT_LEN, MAX_SAMPLES_PER_NODE)
        node_data[node_id] = (ctx, tgt)
        print(f"[server] Node {node_id}: {len(tgt)} sample training siap dikirim")

    eval_context, eval_target = make_samples(eval_text, CONTEXT_LEN, 200)
    print(f"[server] Eval set (holdout, tak pernah dikirim ke node): {len(eval_target)} sample")


def relu(x):
    return np.maximum(x, 0.0)


def softmax_rows(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def evaluate_loss(weights_flat: np.ndarray) -> float:
    offset = 0
    w1 = weights_flat[offset:offset + W1_SIZE].reshape(INPUT_DIM, HIDDEN_DIM); offset += W1_SIZE
    b1 = weights_flat[offset:offset + B1_SIZE]; offset += B1_SIZE
    w2 = weights_flat[offset:offset + W2_SIZE].reshape(HIDDEN_DIM, VOCAB_SIZE); offset += W2_SIZE
    b2 = weights_flat[offset:offset + B2_SIZE]

    x = eval_context.astype(np.float32) / VOCAB_SIZE
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    probs = softmax_rows(z2)

    n = len(eval_target)
    p_correct = probs[np.arange(n), eval_target]
    p_correct = np.clip(p_correct, 1e-7, None)
    return float(np.mean(-np.log(p_correct)))


def make_progress_bar(fraction: float, width: int = 24) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    bar = "█" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction * 100:5.1f}%"


def print_progress(round_num: int, eval_loss: float):
    round_fraction = round_num / MAX_ROUNDS

    # Progress loss: 0% di BASELINE_LOSS (tebak acak), 100% di TARGET_LOSS atau lebih rendah
    if BASELINE_LOSS > TARGET_LOSS:
        loss_fraction = (BASELINE_LOSS - eval_loss) / (BASELINE_LOSS - TARGET_LOSS)
    else:
        loss_fraction = 1.0 if eval_loss <= TARGET_LOSS else 0.0

    print(f"[server] Ronde   {round_num:3d}/{MAX_ROUNDS}  {make_progress_bar(round_fraction)}")
    print(f"[server] Loss    {eval_loss:.4f} -> target {TARGET_LOSS}  {make_progress_bar(loss_fraction)}")


def save_checkpoint(round_num: int, is_final: bool = False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    round_path = os.path.join(CHECKPOINT_DIR, f"round_{round_num}.bin")
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.bin")

    global_weights.tofile(round_path)
    global_weights.tofile(latest_path)

    if is_final:
        final_path = os.path.join(CHECKPOINT_DIR, "final.bin")
        global_weights.tofile(final_path)
        print(f"[server] Checkpoint FINAL disimpan: {final_path}")

    print(f"[server] Checkpoint disimpan: {round_path} (dan latest.bin)")


@app.route("/get_training_data", methods=["GET"])
def get_training_data():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or node_id not in node_data:
        return Response("node_id tidak valid", status=400)

    ctx, tgt = node_data[node_id]
    num_samples = len(tgt)

    payload = struct.pack("<I", num_samples)
    payload += ctx.tobytes()
    payload += tgt.tobytes()

    return Response(payload, mimetype="application/octet-stream")


@app.route("/get_global_weights", methods=["GET"])
def get_global_weights():
    with lock:
        payload = global_weights.tobytes()
    return Response(payload, mimetype="application/octet-stream")


@app.route("/training_status_flag", methods=["GET"])
def training_status_flag():
    with lock:
        flag = 1 if training_complete else 0
    return Response(bytes([flag]), mimetype="application/octet-stream")


@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global global_weights, pending_updates, round_number, training_complete, last_eval_loss

    node_id = request.args.get("node_id", type=int)
    if node_id is None or not (1 <= node_id <= NUM_NODES):
        return Response("node_id tidak valid", status=400)

    with lock:
        if training_complete:
            return Response("TRAINING_COMPLETE", status=200)

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

            last_eval_loss = evaluate_loss(global_weights)
            print(f"[server] === FedAvg ronde #{round_number} selesai ===")
            print_progress(round_number, last_eval_loss)

            is_done = (last_eval_loss <= TARGET_LOSS) or (round_number >= MAX_ROUNDS)
            save_checkpoint(round_number, is_final=is_done)

            if is_done:
                training_complete = True
                reason = "loss mencapai target" if last_eval_loss <= TARGET_LOSS else "mencapai MAX_ROUNDS"
                print(f"[server] *** TRAINING SELESAI ({reason}) — checkpoint final.bin siap dites ***")

    return Response("OK", status=200)


@app.route("/status", methods=["GET"])
def status():
    with lock:
        info = {
            "round_number": round_number,
            "nodes_reported_this_round": list(pending_updates.keys()),
            "param_count": PARAM_COUNT,
            "nodes_with_data": list(node_data.keys()),
            "training_complete": training_complete,
            "last_eval_loss": last_eval_loss,
            "target_loss": TARGET_LOSS,
            "max_rounds": MAX_ROUNDS,
        }
    return info


if __name__ == "__main__":
    print(f"[server] PARAM_COUNT = {PARAM_COUNT}")
    load_and_split_corpus("corpus.txt")
    print(f"[server] TARGET_LOSS={TARGET_LOSS}, MAX_ROUNDS={MAX_ROUNDS}, BASELINE_LOSS={BASELINE_LOSS:.4f}")
    print(f"[server] Menunggu {NUM_NODES} node di /upload_weights")
    app.run(host="0.0.0.0", port=5000, threaded=True)