# src/config.py — konstanta konfigurasi, harus sama persis dengan main.ino di ESP32
import math

CONTEXT_LEN = 8       # sekarang satuannya TOKEN (hasil BPE), bukan karakter lagi
VOCAB_SIZE = 200       # target vocab BPE (vocab_size aktual bisa < ini, dicetak saat training tokenizer)
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LEN
NUM_NODES = 5
MAX_SAMPLES_PER_NODE = 300

PARAM_COUNT = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM) + (HIDDEN_DIM * VOCAB_SIZE + VOCAB_SIZE)
W1_SIZE = INPUT_DIM * HIDDEN_DIM
B1_SIZE = HIDDEN_DIM
W2_SIZE = HIDDEN_DIM * VOCAB_SIZE
B2_SIZE = VOCAB_SIZE

MODEL_DIR = "model"
CORPUS_PATH = "data/corpus.txt"

# ===== KRITERIA BERHENTI TRAINING =====
TARGET_LOSS = 3.0
MAX_ROUNDS = 50
EVAL_HOLDOUT_FRACTION = 0.15

# Baseline loss teoretis kalau model tebak acak (uniform) dari VOCAB_SIZE kelas
BASELINE_LOSS = math.log(VOCAB_SIZE)