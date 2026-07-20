# src/data_loader.py — load corpus, split per node (non-IID), sisakan holdout eval
import numpy as np
from src import config, state


def clean_text(text: str) -> str:
    return "".join(c if ord(c) < config.VOCAB_SIZE else " " for c in text)


def split_corpus(text: str, num_parts: int):
    n = len(text)
    chunk_size = n // num_parts
    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = n if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(text[start:end])
    return chunks


def make_samples(text: str, context_len: int, max_samples: int):
    contexts = []
    targets = []
    for i in range(len(text) - context_len):
        context = text[i:i + context_len]
        target = text[i + context_len]
        contexts.append([ord(c) for c in context])
        targets.append(ord(target))
        if len(contexts) >= max_samples:
            break
    return np.array(contexts, dtype=np.uint8), np.array(targets, dtype=np.uint16)


def load_and_split_corpus(path: str = None):
    path = path or config.CORPUS_PATH

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    text = clean_text(raw)

    split_point = int(len(text) * (1.0 - config.EVAL_HOLDOUT_FRACTION))
    train_text = text[:split_point]
    eval_text = text[split_point:]

    chunks = split_corpus(train_text, config.NUM_NODES)
    for node_id, chunk in enumerate(chunks, start=1):
        ctx, tgt = make_samples(chunk, config.CONTEXT_LEN, config.MAX_SAMPLES_PER_NODE)
        state.node_data[node_id] = (ctx, tgt)
        print(f"[server] Node {node_id}: {len(tgt)} sample training siap dikirim")

    state.eval_context, state.eval_target = make_samples(eval_text, config.CONTEXT_LEN, 200)
    print(f"[server] Eval set (holdout, tak pernah dikirim ke node): {len(state.eval_target)} sample")