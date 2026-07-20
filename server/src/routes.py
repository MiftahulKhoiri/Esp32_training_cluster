# src/routes.py — endpoint HTTP: training data, weight global, upload weight, status
from flask import Blueprint, request, Response
import numpy as np

from src import config, state
from src.model import evaluate_loss
from src.progress import print_progress
from src.checkpoint import save_checkpoint

bp = Blueprint("routes", __name__)


@bp.route("/get_training_data", methods=["GET"])
def get_training_data():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or node_id not in state.node_data:
        return Response("node_id tidak valid", status=400)

    ctx, tgt = state.node_data[node_id]
    num_samples = len(tgt)

    import struct
    payload = struct.pack("<I", num_samples)
    payload += ctx.tobytes()
    payload += tgt.tobytes()

    return Response(payload, mimetype="application/octet-stream")


@bp.route("/get_global_weights", methods=["GET"])
def get_global_weights():
    with state.lock:
        payload = state.global_weights.tobytes()
    return Response(payload, mimetype="application/octet-stream")


@bp.route("/training_status_flag", methods=["GET"])
def training_status_flag():
    with state.lock:
        flag = 1 if state.training_complete else 0
    return Response(bytes([flag]), mimetype="application/octet-stream")


@bp.route("/upload_weights", methods=["POST"])
def upload_weights():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or not (1 <= node_id <= config.NUM_NODES):
        return Response("node_id tidak valid", status=400)

    with state.lock:
        if state.training_complete:
            return Response("TRAINING_COMPLETE", status=200)

    raw = request.get_data()
    expected_bytes = config.PARAM_COUNT * 4
    if len(raw) != expected_bytes:
        print(f"[server] node {node_id}: ukuran payload salah, dapat {len(raw)}, harusnya {expected_bytes}")
        return Response("ukuran payload tidak cocok", status=400)

    weights = np.frombuffer(raw, dtype=np.float32).copy()

    with state.lock:
        state.pending_updates[node_id] = weights
        print(f"[server] Terima weight dari node {node_id} ({len(state.pending_updates)}/{config.NUM_NODES} ronde ini)")

        if len(state.pending_updates) >= config.NUM_NODES:
            stacked = np.stack(list(state.pending_updates.values()), axis=0)
            state.global_weights = np.mean(stacked, axis=0).astype(np.float32)
            state.pending_updates.clear()
            state.round_number += 1

            state.last_eval_loss = evaluate_loss(state.global_weights)
            print(f"[server] === FedAvg ronde #{state.round_number} selesai ===")
            print_progress(state.round_number, state.last_eval_loss)

            is_done = (state.last_eval_loss <= config.TARGET_LOSS) or (state.round_number >= config.MAX_ROUNDS)
            save_checkpoint(state.round_number, is_final=is_done)

            if is_done:
                state.training_complete = True
                reason = "loss mencapai target" if state.last_eval_loss <= config.TARGET_LOSS else "mencapai MAX_ROUNDS"
                print(f"[server] *** TRAINING SELESAI ({reason}) — checkpoint final.bin siap dites ***")

    return Response("OK", status=200)


@bp.route("/status", methods=["GET"])
def status():
    with state.lock:
        info = {
            "round_number": state.round_number,
            "nodes_reported_this_round": list(state.pending_updates.keys()),
            "param_count": config.PARAM_COUNT,
            "nodes_with_data": list(state.node_data.keys()),
            "training_complete": state.training_complete,
            "last_eval_loss": state.last_eval_loss,
            "target_loss": config.TARGET_LOSS,
            "max_rounds": config.MAX_ROUNDS,
        }
    return info