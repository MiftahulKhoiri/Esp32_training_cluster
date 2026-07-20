# src/data_loader.py — load corpus, tokenisasi pakai BPETokenizer C++, split per node
import os
import sys
import numpy as np

from src import config, state

_CPP_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "cpp_tokenizer", "build")
sys.path.append(_CPP_BUILD_DIR)

import bpe_tokenizer_py as bpe  # noqa: E402

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
    return np.array(contexts, dtype=np.uint16), np.array(targets, dtype=np.uint16)


def load_and_split_corpus(path: str = None):
    path = path or config.CORPUS_PATH

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    split_point = int(len(raw_text) * (1.0 - config.EVAL_HOLDOUT_FRACTION))
    train_text = raw_text[:split_point]
    eval_text = raw_text[split_point:]

    tokenizer = bpe.BPETokenizer()
    tokenizer.train(train_text, config.VOCAB_SIZE_TARGET)
    actual_vocab_size = tokenizer.vocab_size()
    print(f"[server] Tokenizer dilatih, vocab_size AKTUAL = {actual_vocab_size} "
          f"(target = {config.VOCAB_SIZE_TARGET})")

    # ===== Set up state runtime berdasar vocab_size AKTUAL, bukan target tetap =====
    state.vocab_size = actual_vocab_size
    state.param_sizes = config.compute_param_sizes(actual_vocab_size)
    state.baseline_loss = config.compute_baseline_loss(actual_vocab_size)
    state.global_weights = np.random.uniform(
        -0.1, 0.1, state.param_sizes["PARAM_COUNT"]
    ).astype(np.float32)
    print(f"[server] PARAM_COUNT (dihitung dari vocab aktual) = {state.param_sizes['PARAM_COUNT']}")

    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)
    tokenizer.save(TOKENIZER_SAVE_PATH)
    print(f"[server] Tokenizer disimpan: {TOKENIZER_SAVE_PATH}")

    state.tokenizer = tokenizer

    chunks = split_corpus(train_text, config.NUM_NODES)
    for node_id, chunk in enumerate(chunks, start=1):
        token_ids = tokenizer.encode(chunk)
        ctx, tgt = make_samples_from_tokens(token_ids, config.CONTEXT_LEN, config.MAX_SAMPLES_PER_NODE)
        state.node_data[node_id] = (ctx, tgt)
        print(f"[server] Node {node_id}: {len(token_ids)} token -> {len(tgt)} sample training siap dikirim")

    eval_token_ids = tokenizer.encode(eval_text)
    state.eval_context, state.eval_target = make_samples_from_tokens(eval_token_ids, config.CONTEXT_LEN, 200)
    print(f"[server] Eval set (holdout, tak pernah dikirim ke node): {len(state.eval_target)} sample")