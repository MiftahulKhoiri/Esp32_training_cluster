# src/monitor.py — cetak ringkasan status tiap beberapa detik, terlepas dari ada
# tidaknya request masuk. Ini yang bikin terminal kelihatan "hidup" real-time,
# bukan cuma diam sampai ada event (weight masuk, FedAvg selesai, dll).
import threading
import time
from src import state
from src.heartbeat import get_active_node_ids

MONITOR_INTERVAL_SEC = 15


def _monitor_loop():
    while True:
        time.sleep(MONITOR_INTERVAL_SEC)

        with state.lock:
            active_ids = get_active_node_ids()
            reported_ids = list(state.pending_updates.keys())
            round_num = state.round_number
            complete = state.training_complete
            elapsed = (time.time() - state.round_start_time) if state.round_start_time else 0
            last_loss = state.last_eval_loss

        if complete:
            print(f"[monitor] Training SELESAI di ronde #{round_num}, loss akhir={last_loss:.4f}")
            continue

        active_str = str(active_ids) if active_ids else "(belum ada node heartbeat)"
        target = len(active_ids) if active_ids else "?"

        print(f"[monitor] Node aktif: {active_str} | "
              f"Ronde #{round_num + 1}: {len(reported_ids)}/{target} sudah lapor weight {reported_ids} | "
              f"elapsed: {elapsed:.0f}s | loss terakhir: {last_loss}")


def start_monitor_thread():
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    print(f"[server] Monitor thread aktif, cetak status tiap {MONITOR_INTERVAL_SEC} detik")