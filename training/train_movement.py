"""Treina o classificador das letras com movimento (H, J, K, X, Z + NADA) a
partir de training/data/movement_sequences.csv (gerado por
collect_movement_data.py) e exporta para
extension/lib/models/movement_classifier.json.

Cada gesto gravado (uma sequência de frames) vira UMA amostra de 60 números,
calculados por movement_features.py — que tem um gêmeo em JavaScript
(extension/src/classifier/movementFeatures.js) para a extensão calcular
exatamente a mesma coisa em tempo real.

Uso:
    python train_movement.py
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from export_model import export
from movement_features import N_FEATURES, extract_features

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "movement_sequences.csv")
MODEL_OUT = os.path.join(
    os.path.dirname(__file__), "..", "extension", "lib", "models", "movement_classifier.json"
)

MIN_SAMPLES_PER_CLASS = 10


def load_sequences():
    """Lê o CSV e devolve (X, y): uma linha de características por gesto."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"{DATA_CSV} não existe — rode collect_movement_data.py primeiro "
            "para gravar os gestos."
        )

    df = pd.read_csv(DATA_CSV)
    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    X, y = [], []
    skipped = 0
    for seq_id, group in df.groupby("sequence_id", sort=False):
        group = group.sort_values("frame")
        label = group["label"].iloc[0]

        coords = group[coord_cols].to_numpy(dtype=np.float64)
        frames = coords.reshape(len(group), 21, 3)
        times = group["t_ms"].to_numpy(dtype=np.float64)

        try:
            X.append(extract_features(frames, times))
            y.append(label)
        except ValueError:
            skipped += 1

    if skipped:
        print(f"{skipped} gesto(s) ignorado(s) por terem frames de menos.")

    return np.array(X, dtype=np.float64), np.array(y)


def main():
    X, y = load_sequences()

    counts = Counter(y)
    print(f"\n{len(X)} gestos, {len(counts)} classes:")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")

    escassas = [f"{label} ({n})" for label, n in counts.items() if n < MIN_SAMPLES_PER_CLASS]
    if escassas:
        raise SystemExit(
            f"\nPoucas amostras em: {', '.join(escassas)}. "
            f"Grave pelo menos {MIN_SAMPLES_PER_CLASS} de cada classe "
            "(o ideal são ~40 por letra e ~60 de NADA) e rode de novo."
        )

    print(f"\nCada gesto virou {X.shape[1]} características (esperado: {N_FEATURES}).")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = MLPClassifier(
        hidden_layer_sizes=(24,),  # menor que o estático: menos classes, menos dados
        activation="relu",
        solver="adam",
        alpha=1e-3,  # regularização um pouco mais forte, o dataset aqui é pequeno
        max_iter=3000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\nAcurácia no conjunto de teste: {acc:.2%}")
    print("\nDesempenho por letra:")
    print(
        classification_report(
            y_test, model.predict(X_test), target_names=encoder.classes_, zero_division=0
        )
    )
    print(
        "Lembre que essa acurácia é medida com os SEUS gestos, gravados na "
        "mesma sessão — o desempenho real tende a ser menor."
    )

    export(model, encoder.classes_, MODEL_OUT)


if __name__ == "__main__":
    main()
