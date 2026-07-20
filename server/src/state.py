# src/state.py — state global; vocab_size & param_sizes baru terisi SETELAH
# tokenizer selesai training (lihat data_loader.load_and_split_corpus)
import threading

lock = threading.Lock()

vocab_size = None       # int, diisi = tokenizer.vocab_size() aktual
param_sizes = None      # dict dari config.compute_param_sizes(vocab_size)
baseline_loss = None    # float, dari config.compute_baseline_loss(vocab_size)

global_weights = None   # np.ndarray, diinisialisasi random SETELAH vocab_size diketahui
pending_updates = {}
round_number = 0
training_complete = False
last_eval_loss = None

node_data = {}
eval_context = None
eval_target = None

tokenizer = None