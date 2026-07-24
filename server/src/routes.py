# src/routes.py — endpoint HTTP: config, kirim matrix B, kirim row block A, terima hasil node
import time
from flask import Blueprint, request, Response
import numpy as np

from src import config, state, storage
from src.matrix_gen import generate_matrices 

bp = Blueprint("routes", __name__)

def get_node_row_info(node_id):
    """Fungsi helper untuk menghitung jangkauan baris per node (menangani sisa bagi)"""
    base_rows = config.N // config.NUM_NODES
    start = (node_id - 1) * base_rows
    
    if node_id == config.NUM_NODES:
        end = config.N
        rows_for_this_node = base_rows + (config.N % config.NUM_NODES)
    else:
        end = start + base_rows
        rows_for_this_node = base_rows
        
    return start, end, rows_for_this_node

@bp.route("/get_config", methods=["GET"])
def get_config():
    # Mengirim config ringkas ke ESP32 agar hemat memori (Format: "N,NUM_NODES")
    config_data = f"{config.N},{config.NUM_NODES}"
    return Response(config_data, status=200, mimetype="text/plain")

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

    start, end, _ = get_node_row_info(node_id)

    with state.lock:
        block = state.matrix_a[start:end, :]
        payload = block.tobytes()

    return Response(payload, mimetype="application/octet-stream")

@bp.route("/submit_result", methods=["POST"])
def submit_result():
    node_id = request.args.get("node_id", type=int)
    if node_id is None or not (1 <= node_id <= config.NUM_NODES):
        return Response("node_id tidak valid", status=400)

    _, _, rows_for_this_node = get_node_row_info(node_id)
    raw = request.get_data()
    
    expected_bytes = rows_for_this_node * config.N * 4
    if len(raw) != expected_bytes:
        print(f"[server] node {node_id}: ukuran payload salah, dapat {len(raw)}, harusnya {expected_bytes}")
        return Response("ukuran payload tidak cocok", status=400)

    block = np.frombuffer(raw, dtype=np.float32).reshape(rows_for_this_node, config.N)

    with state.lock:
        state.results[node_id] = block
        print(f"[server] Hasil node {node_id} diterima ({len(state.results)}/{config.NUM_NODES})")

        if len(state.results) == config.NUM_NODES:
            c_actual = np.vstack([state.results[i] for i in range(1, config.NUM_NODES + 1)])
            max_diff = float(np.max(np.abs(c_actual - state.c_expected)))
            is_correct = max_diff < 1e-3
            elapsed_sec = time.time() - state.round_start_time

            print(f"[server] SEMUA NODE SELESAI. Max diff vs numpy: {max_diff:.6f}, elapsed: {elapsed_sec:.3f}s")
            print("[server] BENAR!" if is_correct else "[server] ADA SELISIH, cek urutan/tipe data")

            storage.save_matrices(config.N, state.matrix_a, state.matrix_b, c_actual, state.c_expected)
            storage.append_log(config.N, config.NUM_NODES, max_diff, is_correct, elapsed_sec)
            
            # Auto-reset state
            state.results.clear()
            print("[server] State di-reset. Menyiapkan ronde baru...")
            generate_matrices()

    return Response("OK", status=200)

@bp.route("/status", methods=["GET"])
def status():
    with state.lock:
        info = {
            "N": config.N,
            "num_nodes": config.NUM_NODES,
            "nodes_reported": list(state.results.keys()),
        }
    return info
