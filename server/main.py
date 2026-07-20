# main.py — entry point server federated learning
from src import config, state
from src.data_loader import load_and_split_corpus
from src.app import create_app

if __name__ == "__main__":
    print(f"[server] PARAM_COUNT = {config.PARAM_COUNT}")
    load_and_split_corpus()  # baca dari config.CORPUS_PATH ("data/corpus.txt")
    print(f"[server] TARGET_LOSS={config.TARGET_LOSS}, MAX_ROUNDS={config.MAX_ROUNDS}, "
          f"BASELINE_LOSS={config.BASELINE_LOSS:.4f}")
    print(f"[server] Menunggu {config.NUM_NODES} node di /upload_weights")

    app = create_app()
    app.run(host="0.0.0.0", port=5000, threaded=True)