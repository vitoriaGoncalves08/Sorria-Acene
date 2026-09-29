"""Treina o classificador de letras a partir de training/data/landmarks.csv
(gerado por collect_data.py) e exporta os pesos para JSON, pronto para a
extensão carregar em extension/lib/models/letter_classifier.json.

Sem TensorFlow de propósito: o modelo é pequeno (63 entradas, uma letra de
saída) e o scikit-learn treina isso sem arrastar as dependências pesadas
(e cheias de conflito de versão) do TensorFlow/TensorFlow.js. O modelo
treinado (pesos de uma rede neural de uma camada escondida) é exportado como
JSON puro, e extension/src/classifier/classifier.js aplica esses pesos com
umas contas simples em JavaScript — sem precisar de nenhuma biblioteca de ML
rodando no navegador.

Uso:
    python train.py                    # só as suas amostras (training/data/landmarks.csv)
    python train.py --with-external     # suas amostras + training/data/external/landmarks_external.csv
                                         # (gerado por process_external_dataset.py; ver training/README.md)
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from export_model import export

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
EXTERNAL_CSV = os.path.join(os.path.dirname(__file__), "data", "external", "landmarks_external.csv")
MODEL_OUT = os.path.join(
    os.path.dirname(__file__), "..", "extension", "lib", "models", "letter_classifier.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-external",
        action="store_true",
        help="também usa training/data/external/landmarks_external.csv no treino (sem alterar os arquivos no disco)",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_CSV)
    if args.with_external:
        if not os.path.exists(EXTERNAL_CSV):
            raise FileNotFoundError(
                f"{EXTERNAL_CSV} não existe — rode process_external_dataset.py primeiro."
            )
        df_external = pd.read_csv(EXTERNAL_CSV)
        df = pd.concat([df, df_external], ignore_index=True)
        print(f"Treinando com {len(df)} amostras (suas + dataset externo).")
    else:
        print(f"Treinando com {len(df)} amostras (só as suas).")

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\nAcurácia no conjunto de teste: {acc:.2%}")

    export(model, encoder.classes_, MODEL_OUT)


if __name__ == "__main__":
    main()
