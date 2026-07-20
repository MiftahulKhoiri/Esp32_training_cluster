# src/data_loader.py — load corpus, tokenisasi pakai BPETokenizer C++, split per node
import os
import sys
import numpy as np

from src import config, state

# Tambah path build C++ tokenizer ke sys.path, lalu import module hasil pybind11
_CPP_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "cpp_tokenizer", "build")
sys.path.append(_CPP_BUILD_DIR)

import bpe_tokenizer_py as bpe  # noqa: E402  (import setelah sys.path.append, sengaja)

TOKENIZER_SAVE_PATH = "model/tokenizer.bin"


def split_corpus(text: str, num_parts: int):
    n = len(text)
    chunk_size = n // num_parts
    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = n if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(text[start:end])
    return chunks


def make_samples_from_tokens(token_ids: list, context_len: int, max_samples: int):
    contexts = []
    targets = []
    for i in range(len(token_ids) - context_len):
        contexts.append(token_ids[i:i + context_len])
        targets.append(token_ids[i + context_len])
        if len(contexts) >= max_samples:
            break
    # uint16 (bukan uint8 lagi) — token id BPE bisa >255 walau kita batasi vocab kecil
    return np.array(contexts, dtype=np.uint16), np.array(targets, dtype=np.uint16)


def load_and_split_corpus(path: str = None):
    path = path or config.CORPUS_PATH

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    split_point = int(len(raw_text) * (1.0 - config.EVAL_HOLDOUT_FRACTION))
    train_text = raw_text[:split_point]
    eval_text = raw_text[split_point:]

    # Latih tokenizer HANYA dari train_text — eval_text tetap benar-benar tidak
    # pernah dilihat, baik oleh node maupun oleh proses training tokenizer-nya sendiri.
    tokenizer = bpe.BPETokenizer()
    tokenizer.train(train_text, config.VOCAB_SIZE)
    actual_vocab_size = tokenizer.vocab_size()
    print(f"[server] Tokenizer dilatih, vocab_size aktual = {actual_vocab_size} "
          f"(target = {config.VOCAB_SIZE})")

    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)
    tokenizer.save(TOKENIZER_SAVE_PATH)
    print(f"[server] Tokenizer disimpan: {TOKENIZER_SAVE_PATH}")

    state.tokenizer = tokenizer  # disimpan di state, siapa tahu dibutuhkan endpoint lain nanti

    # Split train_text per node (kontigu, tetap non-IID), lalu encode tiap potongan
    chunks = split_corpus(train_text, config.NUM_NODES)
    for node_id, chunk in enumerate(chunks, start=1):
        token_ids = tokenizer.encode(chunk)
        ctx, tgt = make_samples_from_tokens(token_ids, config.CONTEXT_LEN, config.MAX_SAMPLES_PER_NODE)
        state.node_data[node_id] = (ctx, tgt)
        print(f"[server] Node {node_id}: {len(token_ids)} token -> {len(tgt)} sample training siap dikirim")

    eval_token_ids = tokenizer.encode(eval_text)
    state.eval_context, state.eval_target = make_samples_from_tokens(eval_token_ids, config.CONTEXT_LEN, 200)
    print(f"[server] Eval set (holdout, tak pernah dikirim ke node): {len(state.eval_target)} sample")