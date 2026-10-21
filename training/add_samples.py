"""Acrescenta à base de treino um CSV baixado da extensão ("Baixar CSV").

Uso:
    python add_samples.py "C:\\Users\\voce\\Downloads\\amostras_confirmadas_123.csv"
    python add_samples.py "C:\\Users\\voce\\Downloads\\gestos_confirmados_123.csv"

Detecta sozinho se o arquivo é de POSES (letras paradas -> data/landmarks.csv)
ou de GESTOS (data/movement_sequences.csv), faz backup do arquivo da base
antes de mexer, e recusa o que estiver errado em vez de sujar a base:

  - colunas diferentes das do CSV de treino;
  - poses na orientação trocada (mão espelhada), que fariam o modelo ver a
    "outra mão" e errar sem dar erro nenhum;
  - linhas que já estão na base (rodar duas vezes o mesmo arquivo não duplica).

Depois de adicionar, rode o treino que o próprio script indica.
"""

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
POSES = os.path.join(HERE, "data", "landmarks.csv")
GESTURES = os.path.join(HERE, "data", "movement_sequences.csv")
MIRROR = np.tile([-1, 1, 1], 21)  # inverte só o x de cada ponto


def feature_key(df):
    return set(map(tuple, df.drop(columns="label").round(6).to_numpy()))


def add_poses(new):
    base = pd.read_csv(POSES)
    if list(new.columns) != list(base.columns):
        sys.exit("As colunas não batem com data/landmarks.csv (p0..p62,label).")

    new = new.assign(label=new.label.astype(str).str.strip())

    # 1) orientação: cada amostra deve se parecer mais com a base como está do
    #    que com a base espelhada
    B = base.drop(columns="label").to_numpy()
    labels = base.label.to_numpy()
    wrong = []
    for i, row in new.iterrows():
        same = B[labels == row.label]
        if len(same) == 0:
            continue
        x = row.drop("label").to_numpy(float)
        if np.linalg.norm(same - x, axis=1).min() > np.linalg.norm(same - x * MIRROR, axis=1).min():
            wrong.append(f"linha {i + 2} ({row.label})")
    if wrong:
        sys.exit(
            "Estas amostras parecem estar com a mão ESPELHADA em relação à sua "
            f"base: {', '.join(wrong)}.\nNada foi adicionado. Isso costuma ser a "
            "outra mão ou um arquivo gravado por uma versão antiga da extensão."
        )

    # 2) duplicatas
    seen = feature_key(base)
    keep = [tuple(r) not in seen for r in new.drop(columns="label").round(6).to_numpy()]
    fresh = new[keep]
    skipped = len(new) - len(fresh)

    if fresh.empty:
        sys.exit("Todas as amostras já estavam na base. Nada a fazer.")
    backup(POSES)
    pd.concat([base, fresh], ignore_index=True).to_csv(POSES, index=False)
    print(f"Poses: +{len(fresh)} adicionadas ({skipped} já existiam). Base: {len(base)} -> {len(base) + len(fresh)}.")
    print("Por letra:", fresh.label.value_counts().sort_index().to_dict())
    print("\nAgora treine:\n  .venv\\Scripts\\python.exe train.py --with-external")


def add_gestures(new):
    base = pd.read_csv(GESTURES)
    if list(new.columns) != list(base.columns):
        sys.exit("As colunas não batem com data/movement_sequences.csv.")

    clash = set(new.sequence_id) & set(base.sequence_id)
    if clash:
        # mesmo id de gravação já na base: dois lotes baixados no mesmo instante
        # seriam fundidos num gesto só — melhor recusar do que corromper
        sys.exit(f"{len(clash)} gesto(s) com sequence_id já existente na base. Nada foi adicionado.")

    backup(GESTURES)
    pd.concat([base, new], ignore_index=True).to_csv(GESTURES, index=False)
    per = new.groupby("sequence_id").label.first().str.strip().value_counts().sort_index().to_dict()
    total = pd.read_csv(GESTURES).groupby("sequence_id").label.first().str.strip().value_counts().sort_index().to_dict()
    print(f"Gestos: +{new.sequence_id.nunique()} adicionados {per}.")
    print("Base agora:", total)
    print("\nAgora treine:\n  .venv\\Scripts\\python.exe train_movement.py")


def backup(path):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = path.replace(".csv", f".backup-{stamp}.csv")
    shutil.copy(path, dest)
    print(f"Backup: {os.path.relpath(dest, HERE)}")


def main():
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        sys.exit(__doc__)
    new = pd.read_csv(sys.argv[1])
    if "sequence_id" in new.columns:
        add_gestures(new)
    elif "p0" in new.columns and "label" in new.columns:
        add_poses(new)
    else:
        sys.exit("Não reconheço este CSV (nem poses, nem gestos).")


if __name__ == "__main__":
    main()
