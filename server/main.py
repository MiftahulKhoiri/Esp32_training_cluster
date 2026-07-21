# main.py — entry point server federated learning
from src import config, state
from src.data_loader import load_and_split_corpus
from src.app import create_app
from src.monitor import start_monitor_thread

if __name__ == "__main__":
    print("[server] Memuat corpus & melatih tokenizer...")
    load_and_split_corpus()

    print(f"[server] vocab_size AKTUAL = {state.vocab_size}, "
          f"PARAM_COUNT = {state.param_sizes['PARAM_COUNT']}")
    print(f"[server] TARGET_LOSS={config.TARGET_LOSS}, MAX_ROUNDS={config.MAX_ROUNDS}, "
          f"BASELINE_LOSS={state.baseline_loss:.4f}")
    print(f"[server] Menunggu {config.NUM_NODES} node di /upload_weights")

    start_monitor_thread()

    app = create_app()
    app.run(host="0.0.0.0", port=5000, threaded=True)