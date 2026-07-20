# src/state.py — state global bersama, dimutasi langsung lewat module attribute
# (bukan pakai keyword `global` lintas modul — assignment ke state.xxx sudah cukup)
import threading
import numpy as np
from src import config

lock = threading.Lock()

global_weights = np.random.uniform(-0.1, 0.1, config.PARAM_COUNT).astype(np.float32)
pending_updates = {}   # node_id -> np.array weight
round_number = 0
training_complete = False
last_eval_loss = None

node_data = {}          # node_id -> (context_ids, target_ids)
eval_context = None     # (N, CONTEXT_LEN) uint8 — holdout, tak pernah dikirim ke node
eval_target = None      # (N,) uint16