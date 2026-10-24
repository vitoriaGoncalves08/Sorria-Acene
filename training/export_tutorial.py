"""Escolhe, para cada letra, UMA gravação sua para servir de exemplo no
tutorial da extensão (o pop-up do pinguim que abre ao clicar numa letra do
alfabeto) e salva em extension/lib/models/tutorial_refs.json.

A escolhida é a mais "típica" da letra: a amostra mais próxima da média de
todas as amostras dela. Assim uma gravação torta ou cortada dificilmente vira
o exemplo.

Só usa os SEUS dados (landmarks.csv e movement_sequences.csv), nunca o
dataset externo: o tutorial diz "referência gravada pelo autor", e isso
precisa ser verdade.

Uso:
    python export_tutorial.py

Rode de novo sempre que gravar mais dados, para o exemplo acompanhar.
"""

import json
import os

import numpy as np
import pandas as pd

from movement_features import extract_features

HERE = os.path.dirname(__file__)
POSES_CSV = os.path.join(HERE, "data", "landmarks.csv")
GESTURES_CSV = os.path.join(HERE, "data", "movement_sequences.csv")
OUT = os.path.join(HERE, "..", "extension", "lib", "models", "tutorial_refs.json")

NONE_LABEL = "NADA"  # movimento à toa: não é letra, não tem tutorial
FRAME_MS = 1000.0 / 30.0  # o gesto é salvo a 30 quadros/s: arquivo pequeno e
# animação igual, venha a gravação de 30 ou 60 quadros/s
DECIMALS = 4


def medoid_index(X):
    """Índice da linha mais próxima da média das linhas."""
    return int(np.argmin(np.linalg.norm(X - X.mean(axis=0), axis=1)))


def export_poses():
    df = pd.read_csv(POSES_CSV)
    cols = [f"p{i}" for i in range(63)]
    poses = {}
    for label, group in df.groupby("label"):
        X = group[cols].to_numpy(dtype=np.float64)
        pose = X[medoid_index(X)].reshape(21, 3)[:, :2]  # o z não é desenhado
        poses[label] = np.round(pose, DECIMALS).tolist()
        print(f"  pose {label}: escolhida entre {len(X)} amostras")
    return poses


def resample(frames, times):
    """Reamostra o gesto a FRAME_MS, com tempos começando em zero."""
    t = times - times[0]
    new_t = np.arange(0.0, t[-1] + 1e-6, FRAME_MS)
    flat = frames.reshape(len(frames), -1)
    out = np.stack([np.interp(new_t, t, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape(len(new_t), 21, 2)


def export_gestures():
    df = pd.read_csv(GESTURES_CSV)
    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    by_label = {}
    for seq_id, group in df.groupby("sequence_id", sort=False):
        label = group["label"].iloc[0]
        if label == NONE_LABEL:
            continue
        group = group.sort_values("frame")
        frames = group[coord_cols].to_numpy(dtype=np.float64).reshape(len(group), 21, 3)
        times = group["t_ms"].to_numpy(dtype=np.float64)
        try:
            features = extract_features(frames, times)
        except ValueError:
            continue  # frames de menos
        by_label.setdefault(label, []).append((seq_id, features, frames[:, :, :2], times))

    if not by_label:
        return {}

    # As características têm escalas bem diferentes (forma da mão, trajeto,
    # duração...). Sem padronizar, a "média" seria dominada por uma só delas.
    all_features = np.array([f for items in by_label.values() for _, f, _, _ in items])
    mean, std = all_features.mean(axis=0), all_features.std(axis=0) + 1e-9

    gestures = {}
    for label, items in sorted(by_label.items()):
        X = np.array([(f - mean) / std for _, f, _, _ in items])
        seq_id, _, frames, times = items[medoid_index(X)]
        resampled = resample(frames, times)
        gestures[label] = {
            "frame_ms": round(FRAME_MS, 3),
            "frames": np.round(resampled, DECIMALS).tolist(),
        }
        duration = (times[-1] - times[0]) / 1000
        print(f"  gesto {label}: sequência {seq_id} ({duration:.2f} s), escolhida entre {len(items)}")
    return gestures


def main():
    print("Escolhendo os exemplos do tutorial...")
    refs = {
        "source": "gravações do autor do projeto (não é material oficial de Libras)",
        "poses": export_poses() if os.path.exists(POSES_CSV) else {},
        "gestures": export_gestures() if os.path.exists(GESTURES_CSV) else {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = os.path.getsize(OUT) / 1024
    print(
        f"\n{len(refs['poses'])} poses e {len(refs['gestures'])} gestos salvos em "
        f"{os.path.normpath(OUT)} ({size_kb:.0f} KB). Recarregue a extensão."
    )


if __name__ == "__main__":
    main()
