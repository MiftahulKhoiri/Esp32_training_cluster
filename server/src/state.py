# src/state.py — state global; vocab_size & param_sizes baru terisi SETELAH
# tokenizer selesai training (lihat data_loader.load_and_split_corpus)
import threading

lock = threading.Lock()

vocab_size = None
param_sizes = None
baseline_loss = None

global_weights = None
pending_updates = {}
round_number = 0
round_start_time = None   # timestamp mulai kumpul weight ronde ini, None kalau belum ada yang lapor
training_complete = False
last_eval_loss = None

node_data = {}
eval_context = None
eval_target = None

tokenizer = None

active_nodes = {}   # node_id -> timestamp heartbeat/lapor terakhir