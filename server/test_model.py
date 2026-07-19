# test_model.py — load checkpoint hasil federated training, tes prediksi karakter berikutnya
# Cara pakai:
#   python test_model.py checkpoints/latest.bin "A: Pagi! U"
import sys
import numpy as np

# ===== HARUS SAMA PERSIS DENGAN main.ino & server.py =====
CONTEXT_LEN = 8
VOCAB_SIZE = 128
HIDDEN_DIM = 64
INPUT_DIM = CONTEXT_LEN

W1_SIZE = INPUT_DIM * HIDDEN_DIM
B1_SIZE = HIDDEN_DIM
W2_SIZE = HIDDEN_DIM * VOCAB_SIZE
B2_SIZE = VOCAB_SIZE
PARAM_COUNT = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE


def load_checkpoint(path: str):
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size != PARAM_COUNT:
        raise ValueError(f"Ukuran checkpoint tidak cocok: dapat {flat.size}, harusnya {PARAM_COUNT}")

    offset = 0
    w1 = flat[offset:offset + W1_SIZE].reshape(INPUT_DIM, HIDDEN_DIM); offset += W1_SIZE
    b1 = flat[offset:offset + B1_SIZE]; offset += B1_SIZE
    w2 = flat[offset:offset + W2_SIZE].reshape(HIDDEN_DIM, VOCAB_SIZE); offset += W2_SIZE
    b2 = flat[offset:offset + B2_SIZE]; offset += B2_SIZE

    return w1, b1, w2, b2


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def forward(context_text: str, w1, b1, w2, b2):
    # Sama persis dengan encoding di main.ino: id char / VOCAB_SIZE, panjang CONTEXT_LEN
    if len(context_text) < CONTEXT_LEN:
        context_text = context_text.rjust(CONTEXT_LEN)  # padding spasi di depan
    context_text = context_text[-CONTEXT_LEN:]  # ambil CONTEXT_LEN karakter terakhir

    x = np.array([ord(c) / VOCAB_SIZE for c in context_text], dtype=np.float32)

    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    probs = softmax(z2)

    return probs, context_text


def print_top_k(probs, k=5):
    top_idx = np.argsort(probs)[::-1][:k]
    print(f"  Top-{k} prediksi karakter berikutnya:")
    for idx in top_idx:
        ch = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
        print(f"    '{ch}'  (id={idx})  prob={probs[idx]:.4f}")


def generate(context_text: str, w1, b1, w2, b2, length=30):
    result = context_text
    for _ in range(length):
        probs, _ = forward(result, w1, b1, w2, b2)
        next_id = int(np.argmax(probs))  # greedy, bukan sampling
        next_char = chr(next_id) if 32 <= next_id < 127 else " "
        result += next_char
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cara pakai: python test_model.py <checkpoint.bin> \"<teks_konteks>\"")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    context_text = sys.argv[2]

    w1, b1, w2, b2 = load_checkpoint(checkpoint_path)

    probs, used_context = forward(context_text, w1, b1, w2, b2)
    print(f"Konteks dipakai (setelah padding/potong ke {CONTEXT_LEN} char): '{used_context}'")
    print_top_k(probs)

    print()
    generated = generate(context_text, w1, b1, w2, b2, length=30)
    print(f"Generate greedy 30 karakter: '{generated}'")