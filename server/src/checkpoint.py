# src/checkpoint.py — simpan model gabungan (hasil FedAvg) ke folder model/, nama auto-increment
import os
import re
from src import config, state

MODEL_DIR = "model"
MODEL_NAME_PATTERN = re.compile(r"^model(\d+)\.bin$")


def _next_model_path() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)

    existing_numbers = []
    for filename in os.listdir(MODEL_DIR):
        match = MODEL_NAME_PATTERN.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = (max(existing_numbers) + 1) if existing_numbers else 1
    return os.path.join(MODEL_DIR, f"model{next_number}.bin")


def save_model(is_final: bool = False) -> str:
    path = _next_model_path()
    state.global_weights.tofile(path)

    if is_final:
        print(f"[server] Model FINAL disimpan: {path}")
    else:
        print(f"[server] Model disimpan: {path}")

    return path