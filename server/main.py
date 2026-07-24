# main.py — entry point komputasi paralel matmul
from src.matrix_gen import generate_matrices
from src.app import create_app

if __name__ == "__main__":
    generate_matrices()
    app = create_app()
    app.run(host="0.0.0.0", port=5000)