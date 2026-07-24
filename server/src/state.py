# src/state.py — state global: matriks A, B, dan hasil yang lapor dari tiap node
import threading

lock = threading.Lock()

matrix_a = None      # np.ndarray (N, N)
matrix_b = None      # np.ndarray (N, N)
c_expected = None    # np.ndarray (N, N) — referensi hasil A @ B via numpy

results = {}         # node_id -> np.ndarray (ROWS_PER_NODE, N)