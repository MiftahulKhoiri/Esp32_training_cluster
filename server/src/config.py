# src/config.py — konfigurasi komputasi paralel matmul
N = 10                  # ukuran matriks persegi — naikkan bertahap: 10, 25, 50, 75, 100, 150, 200...
NUM_NODES = 5
ROWS_PER_NODE = N // NUM_NODES
RANDOM_SEED = 42