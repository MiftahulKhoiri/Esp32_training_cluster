# src/routes.py — endpoint HTTP: kirim matrix B, kirim row block A, terima hasil node
from flask import Blueprint, request, Response
import numpy as np

from src import config, state

bp = Blueprint("routes", __name__)


@bp.route("/get_matrix_b", methods=["GET"])
def get_matrix_b():
    with state.lock:
        payload = state.matrix_b.tobytes()
    return Response(payload, mimetype="application/octet-stream")


@bp.route("/get_row_block", methods=["GET"])
def get_row_block():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or not (1 <= node_id <= config.NUM_NODES):
        return Response("node_id tidak valid", status=400)

    start = (node_id - 1) * config.ROWS_PER_NODE
    end = start + config.ROWS_PER_NODE

    with state.lock:
        block = state.matrix_a[start:end, :]
        payload = block.tobytes()

    return Response(payload, mimetype="application/octet-stream")


@bp.route("/submit_result", methods=["POST"])
def submit_result():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or not (1 <= node_id <= config.NUM_NODES):
        return Response("node_id tidak valid", status=400)

    raw = request.get_data()
    expected_bytes = config.ROWS_PER_NODE * config.N * 4
    if len(raw) != expected_bytes:
        print(f"[server] node {node_id}: ukuran payload salah, dapat {len(raw)}, harusnya {expected_bytes}")
        return Response("ukuran payload tidak cocok", status=400)

    block = np.frombuffer(raw, dtype=np.float32).reshape(config.ROWS_PER_NODE, config.N)

    with state.lock:
        state.results[node_id] = block
        print(f"[server] Hasil node {node_id} diterima ({len(state.results)}/{config.NUM_NODES})")

        if len(state.results) == config.NUM_NODES:
            c_actual = np.vstack([state.results[i] for i in range(1, config.NUM_NODES + 1)])
            max_diff = float(np.max(np.abs(c_actual - state.c_expected)))
            print(f"[server] SEMUA NODE SELESAI. Max diff vs numpy: {max_diff:.6f}")
            print("[server] BENAR!" if max_diff < 1e-3 else "[server] ADA SELISIH, cek urutan/tipe data")

    return Response("OK", status=200)


@bp.route("/status", methods=["GET"])
def status():
    with state.lock:
        info = {
            "N": config.N,
            "num_nodes": config.NUM_NODES,
            "rows_per_node": config.ROWS_PER_NODE,
            "nodes_reported": list(state.results.keys()),
        }
    return info