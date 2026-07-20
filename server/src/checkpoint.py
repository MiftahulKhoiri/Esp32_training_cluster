# src/checkpoint.py — simpan global_weights ke file, termasuk penanda final
import os
from src import config, state


def save_checkpoint(round_num: int, is_final: bool = False):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    round_path = os.path.join(config.CHECKPOINT_DIR, f"round_{round_num}.bin")
    latest_path = os.path.join(config.CHECKPOINT_DIR, "latest.bin")

    state.global_weights.tofile(round_path)
    state.global_weights.tofile(latest_path)

    if is_final:
        final_path = os.path.join(config.CHECKPOINT_DIR, "final.bin")
        state.global_weights.tofile(final_path)
        print(f"[server] Checkpoint FINAL disimpan: {final_path}")

    print(f"[server] Checkpoint disimpan: {round_path} (dan latest.bin)")