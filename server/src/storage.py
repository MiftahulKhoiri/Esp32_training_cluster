# src/storage.py — simpan matriks & log hasil uji ke file supaya bisa dibaca/diolah lagi
import os
import csv
import time
import numpy as np
from src import config


def ensure_dirs():
    os.makedirs(config.MATRIX_DIR, exist_ok=True)


def save_matrices(N, matrix_a, matrix_b, c_actual, c_expected):
    ensure_dirs()
    np.save(os.path.join(config.MATRIX_DIR, f"A_N{N}.npy"), matrix_a)
    np.save(os.path.join(config.MATRIX_DIR, f"B_N{N}.npy"), matrix_b)
    np.save(os.path.join(config.MATRIX_DIR, f"C_actual_N{N}.npy"), c_actual)
    np.save(os.path.join(config.MATRIX_DIR, f"C_expected_N{N}.npy"), c_expected)
    print(f"[server] Matriks N={N} disimpan ke {config.MATRIX_DIR}/")


def append_log(N, num_nodes, max_diff, is_correct, elapsed_sec):
    ensure_dirs()
    file_exists = os.path.isfile(config.LOG_PATH)
    with open(config.LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "N", "num_nodes", "max_diff", "correct", "elapsed_sec"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            N, num_nodes, f"{max_diff:.6f}", is_correct, f"{elapsed_sec:.3f}"
        ])
    print(f"[server] Log ditambahkan ke {config.LOG_PATH}")