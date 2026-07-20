# src/routes.py — endpoint HTTP: heartbeat, vocab_size, training data, weight, status
from flask import Blueprint, request, Response
import numpy as np
import struct
import time

from src import config, state
from src.model import evaluate_loss
from src.progress import print_progress
from src.checkpoint import save_model
from src.heartbeat import record_heartbeat, count_active_nodes, get_active_node_ids

bp = Blueprint("routes", __name__)


@bp.route("/heartbeat", methods=["GET"])
def heartbeat():
    node_id = request.args.get("node_id", type=int)
    if node_id is None:
        return Response("node_id tidak valid", status=400)
    record_heartbeat(node_id)
    return Response("OK", status=200)


@bp.route("/vocab_size", methods=["GET"])
def vocab_size_endpoint():
    with state.lock:
        vs = state.vocab_size if state.vocab_size is not None else 0
    payload = struct.pack("<H", vs)
    return Response(payload, mimetype="application/octet-stream")


@bp.route("/get_training_data", methods=["GET"])
def get_training_data():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or node_id not in state.node_data:
        return Response("node_id tidak valid", status=400)

    ctx, tgt = state.node_data[node_id]
    num_samples = len(tgt)

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
    expected_bytes = state.param_sizes["PARAM_COUNT"] * 4
    if len(raw) != expected_bytes:
        print(f"[server] node {node_id}: ukuran payload salah, dapat {len(raw)}, harusnya {expected_bytes}")
        return Response("ukuran payload tidak cocok", status=400)

    weights = np.frombuffer(raw, dtype=np.float32).copy()

    with state.lock:
        if len(state.pending_updates) == 0:
            state.round_start_time = time.time()

        state.pending_updates[node_id] = weights
        active_count = count_active_nodes()
        elapsed = time.time() - state.round_start_time

        print(f"[server] Terima weight dari node {node_id} "
              f"({len(state.pending_updates)}/{active_count} node aktif lapor, "
              f"elapsed={elapsed:.0f}s)")

        should_aggregate = False
        trigger_reason = ""

        if len(state.pending_updates) >= max(active_count, 1):
            should_aggregate = True
            trigger_reason = "semua node aktif sudah lapor"
        elif elapsed >= config.ROUND_TIMEOUT_SEC and len(state.pending_updates) >= config.MIN_NODES_FOR_FEDAVG:
            should_aggregate = True
            trigger_reason = f"timeout ronde ({config.ROUND_TIMEOUT_SEC}s)"

        if should_aggregate:
            stacked = np.stack(list(state.pending_updates.values()), axis=0)
            state.global_weights = np.mean(stacked, axis=0).astype(np.float32)
            n_participated = len(state.pending_updates)
            state.pending_updates.clear()
            state.round_start_time = None
            state.round_number += 1

            state.last_eval_loss = evaluate_loss(state.global_weights)
            print(f"[server] === FedAvg ronde #{state.round_number} selesai "
                  f"({n_participated} node ikut, dipicu: {trigger_reason}) ===")
            print_progress(state.round_number, state.last_eval_loss)

            is_done = (state.last_eval_loss <= config.TARGET_LOSS) or (state.round_number >= config.MAX_ROUNDS)
            saved_path = save_model(is_final=is_done)

            if is_done:
                state.training_complete = True
                reason = "loss mencapai target" if state.last_eval_loss <= config.TARGET_LOSS else "mencapai MAX_ROUNDS"
                print(f"[server] *** TRAINING SELESAI ({reason}) — model final: {saved_path} ***")

    return Response("OK", status=200)


@bp.route("/status", methods=["GET"])
def status():
    with state.lock:
        info = {
            "round_number": state.round_number,
            "nodes_reported_this_round": list(state.pending_updates.keys()),
            "active_node_ids": get_active_node_ids(),
            "active_node_count": count_active_nodes(),
            "vocab_size": state.vocab_size,
            "param_count": state.param_sizes["PARAM_COUNT"] if state.param_sizes else None,
            "nodes_with_data": list(state.node_data.keys()),
            "training_complete": state.training_complete,
            "last_eval_loss": state.last_eval_loss,
            "target_loss": config.TARGET_LOSS,
            "max_rounds": config.MAX_ROUNDS,
        }
    return info