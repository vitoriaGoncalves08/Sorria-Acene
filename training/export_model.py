"""Exporta um MLPClassifier do scikit-learn como JSON puro, no formato que
extension/src/classifier/mlp.js sabe aplicar.

Usado tanto pelo treino das letras paradas (train.py) quanto pelo das letras
com movimento (train_movement.py) — o formato é o mesmo, só mudam as entradas
e as classes.
"""

import json
import os


def export(model, labels, output_path):
    # coefs_[i] tem shape (n_entradas_da_camada, n_saídas_da_camada) e já está
    # na ordem certa para o forward pass em mlp.js (out[j] = soma(in[i] *
    # W[i][j]) + b[j]).
    layers = []
    for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        is_last = i == len(model.coefs_) - 1
        layers.append(
            {
                "weights": weights.tolist(),
                "biases": biases.tolist(),
                "activation": "softmax" if is_last else "relu",
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"labels": list(labels), "layers": layers}, f)

    print(f"\nModelo exportado para: {os.path.abspath(output_path)}")
