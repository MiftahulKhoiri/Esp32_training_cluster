# src/config.py — konstanta tetap + fungsi hitung ukuran param dari vocab_size aktual
import math

CONTEXT_LEN = 8              # satuan TOKEN BPE, bukan karakter
VOCAB_SIZE_TARGET = 200      # target training tokenizer — vocab AKTUAL ada di state.vocab_size
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LEN
NUM_NODES = 5
MAX_SAMPLES_PER_NODE = 300

MODEL_DIR = "model"
CORPUS_PATH = "data/corpus.txt"

TARGET_LOSS = 3.0
MAX_ROUNDS = 50
EVAL_HOLDOUT_FRACTION = 0.15

# Tambahan di src/config.py, setelah EVAL_HOLDOUT_FRACTION:

# ===== HEARTBEAT & TIMEOUT (deteksi node aktif) =====
HEARTBEAT_TIMEOUT_SEC = 90     # node dianggap MATI kalau tidak heartbeat dalam waktu ini
ROUND_TIMEOUT_SEC = 120        # maksimal tunggu 1 ronde sebelum FedAvg dipaksa jalan
MIN_NODES_FOR_FEDAVG = 2       # minimal node yang harus lapor sebelum timeout boleh trigger FedAvg

def compute_param_sizes(vocab_size: int) -> dict:
    """Ukuran parameter model, dihitung dari vocab_size AKTUAL (bukan target tetap)."""
    w1_size = INPUT_DIM * HIDDEN_DIM
    b1_size = HIDDEN_DIM
    w2_size = HIDDEN_DIM * vocab_size
    b2_size = vocab_size
    return {
        "W1_SIZE": w1_size,
        "B1_SIZE": b1_size,
        "W2_SIZE": w2_size,
        "B2_SIZE": b2_size,
        "PARAM_COUNT": w1_size + b1_size + w2_size + b2_size,
    }


def compute_baseline_loss(vocab_size: int) -> float:
    return math.log(vocab_size)