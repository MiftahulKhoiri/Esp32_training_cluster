# src/heartbeat.py — lacak node mana yang masih hidup lewat heartbeat/last-seen
#
# PENTING: fungsi di sini TIDAK mengambil state.lock sendiri — caller wajib
# sudah berada di dalam `with state.lock:` sebelum manggil ini. Ini supaya
# aman dipanggil dari dalam blok lock lain (mis. /status, /upload_weights)
# tanpa deadlock. record_heartbeat() adalah pengecualian (dia entry point
# independen, jadi dia pegang lock sendiri).
import time
from src import config, state


def record_heartbeat(node_id: int):
    with state.lock:
        state.active_nodes[node_id] = time.time()


def count_active_nodes() -> int:
    """WAJIB dipanggil dari dalam `with state.lock:` di caller."""
    now = time.time()
    return sum(
        1 for last_seen in state.active_nodes.values()
        if now - last_seen <= config.HEARTBEAT_TIMEOUT_SEC
    )


def get_active_node_ids() -> list:
    """WAJIB dipanggil dari dalam `with state.lock:` di caller."""
    now = time.time()
    return sorted(
        node_id for node_id, last_seen in state.active_nodes.items()
        if now - last_seen <= config.HEARTBEAT_TIMEOUT_SEC
    )