# data_prep.py — split corpus percakapan jadi 5 bagian + char-level encoding
# Jalankan sekali di Pi/laptop, hasilnya di-copy ke tiap folder sketch ESP32.

VOCAB_SIZE = 128       # harus sama dengan main.ino & server.py
CONTEXT_LEN = 8         # harus sama dengan main.ino & server.py
NUM_NODES = 5
MAX_SAMPLES_PER_NODE = 300  # batasi biar Matrix di ESP32 tetap kecil di RAM

def clean_text(text: str) -> str:
    # Buang karakter non-ASCII (di luar VOCAB_SIZE), ganti spasi
    return "".join(c if ord(c) < VOCAB_SIZE else " " for c in text)

def split_corpus(text: str, num_parts: int):
    # Split kontigu (bukan acak) -> tiap node dapat "topik"/gaya beda -> non-IID asli
    n = len(text)
    chunk_size = n // num_parts
    chunks = []
    for i in range(num_parts):
        start = i * chunk_size
        end = n if i == num_parts - 1 else (i + 1) * chunk_size
        chunks.append(text[start:end])
    return chunks

def make_samples(text: str, context_len: int, max_samples: int):
    samples = []
    for i in range(len(text) - context_len):
        context = text[i:i + context_len]
        target = text[i + context_len]
        context_ids = [ord(c) for c in context]
        target_id = ord(target)
        samples.append((context_ids, target_id))
        if len(samples) >= max_samples:
            break
    return samples

def write_header(node_id: int, samples: list, out_dir: str = "."):
    n = len(samples)
    lines = []
    lines.append(f"// data_node{node_id}.h — auto-generated oleh data_prep.py, JANGAN edit manual")
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append("#include <cstddef>")
    lines.append("")
    lines.append(f"const size_t NODE_NUM_SAMPLES = {n};")
    lines.append(f"const uint8_t NODE_CONTEXT_IDS[{n}][{CONTEXT_LEN}] = {{")
    for ctx, _ in samples:
        lines.append("  {" + ", ".join(str(x) for x in ctx) + "},")
    lines.append("};")
    lines.append("")
    lines.append(f"const uint16_t NODE_TARGET_IDS[{n}] = {{")
    lines.append("  " + ", ".join(str(t) for _, t in samples))
    lines.append("};")

    path = f"{out_dir}/data_node{node_id}.h"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[data_prep] {path} ditulis, {n} sample")


if __name__ == "__main__":
    with open("corpus.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    text = clean_text(raw)
    chunks = split_corpus(text, NUM_NODES)

    for node_id, chunk in enumerate(chunks, start=1):
        samples = make_samples(chunk, CONTEXT_LEN, MAX_SAMPLES_PER_NODE)
        write_header(node_id, samples, out_dir=f"node{node_id}_data")