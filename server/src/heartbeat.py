# src/heartbeat.py — lacak node mana yang masih hidup lewat heartbeat/last-seen
import time
from src import config, state


def record_heartbeat(node_id: int):
    with state.lock:
        state.active_nodes[node_id] = time.time()


def count_active_nodes() -> int:
    now = time.time()
    with state.lock:
        return sum(
            1 for last_seen in state.active_nodes.values()
            if now - last_seen <= config.HEARTBEAT_TIMEOUT_SEC
        )


def get_active_node_ids() -> list:
    now = time.time()
    with state.lock:
        return sorted(
            node_id for node_id, last_seen in state.active_nodes.items()
            if now - last_seen <= config.HEARTBEAT_TIMEOUT_SEC
        )