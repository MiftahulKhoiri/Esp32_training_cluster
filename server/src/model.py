# src/model.py — forward pass numpy, replikasi persis dense_layer.h di ESP32
import numpy as np
from src import config, state


def relu(x):
    return np.maximum(x, 0.0)


def softmax_rows(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def evaluate_loss(weights_flat: np.ndarray) -> float:
    sizes = state.param_sizes
    offset = 0
    w1 = weights_flat[offset:offset + sizes["W1_SIZE"]].reshape(config.INPUT_DIM, config.HIDDEN_DIM)
    offset += sizes["W1_SIZE"]
    b1 = weights_flat[offset:offset + sizes["B1_SIZE"]]
    offset += sizes["B1_SIZE"]
    w2 = weights_flat[offset:offset + sizes["W2_SIZE"]].reshape(config.HIDDEN_DIM, state.vocab_size)
    offset += sizes["W2_SIZE"]
    b2 = weights_flat[offset:offset + sizes["B2_SIZE"]]

    x = state.eval_context.astype(np.float32) / state.vocab_size
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    probs = softmax_rows(z2)

    n = len(state.eval_target)
    p_correct = probs[np.arange(n), state.eval_target]
    p_correct = np.clip(p_correct, 1e-7, None)
    return float(np.mean(-np.log(p_correct)))