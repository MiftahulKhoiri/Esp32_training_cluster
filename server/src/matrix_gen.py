# src/matrix_gen.py — generate matriks A, B, dan hasil referensi numpy saat startup
import time
import numpy as np
from src import config, state


def generate_matrices():
    np.random.seed(config.RANDOM_SEED)
    state.matrix_a = np.random.uniform(-1, 1, size=(config.N, config.N)).astype(np.float32)
    state.matrix_b = np.random.uniform(-1, 1, size=(config.N, config.N)).astype(np.float32)
    state.c_expected = state.matrix_a @ state.matrix_b
    state.round_start_time = time.time()
    print(f"[server] Matriks A dan B ({config.N}x{config.N}) dibuat, "
          f"{config.NUM_NODES} node, {config.ROWS_PER_NODE} baris/node")