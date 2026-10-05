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
        weights, biases = weights.tolist(), biases.tolist()

        # Com exatamente 2 classes, o scikit-learn usa UMA saída só (sigmoide),
        # não duas. Sem tratar isso, o softmax do mlp.js recebe um único valor
        # e devolve sempre 100% para a primeira classe — o modelo parece
        # funcionar no treino e responde a mesma coisa para tudo na extensão.
        # sigmoid(z) == softmax([0, z])[1], então viramos 2 saídas equivalentes.
        if is_last and len(biases) == 1:
            weights = [[0.0, w[0]] for w in weights]
            biases = [0.0, biases[0]]

        layers.append(
            {
                "weights": weights,
                "biases": biases,
                "activation": "softmax" if is_last else "relu",
            }
        )

    # trava de segurança: uma saída por classe, senão o modelo erra em silêncio
    n_outputs = len(layers[-1]["biases"])
    if n_outputs != len(labels):
        raise ValueError(
            f"O modelo tem {n_outputs} saídas, mas há {len(labels)} classes "
            f"({list(labels)}). Exportar assim geraria um modelo que erra em silêncio."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"labels": list(labels), "layers": layers}, f)

    print(f"\nModelo exportado para: {os.path.abspath(output_path)}")
