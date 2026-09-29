"""Verifica que movement_features.py (Python) e movementFeatures.js
(JavaScript) calculam EXATAMENTE as mesmas características.

Se os dois divergirem, o modelo treinado aqui recebe entradas diferentes na
extensão e passa a errar sem dar nenhum erro visível — o tipo de bug mais
chato de achar. Por isso este teste existe.

Uso:
    python check_parity.py

Gera training/data/parity_fixture.json com um gesto sintético e as
características esperadas. Para conferir o lado JavaScript, abra a aba da
extensão no Chrome, abra o console (F12) e cole o conteúdo de
extension/src/classifier/parityCheck.js.
"""

import json
import math
import os

from movement_features import N_FEATURES, extract_features

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "data", "parity_fixture.json")


def synthetic_gesture(n_frames=24):
    """Um gesto determinístico e reprodutível: a mão faz um zigue-zague (como
    um Z) enquanto os dedos se movem um pouco. Sem aleatoriedade, para os dois
    lados poderem comparar número a número.
    """
    frames = []
    times = []
    for f in range(n_frames):
        t = f / (n_frames - 1)  # 0 -> 1
        # zigue-zague no eixo x, descendo no y
        wander = 0.10 * math.sin(t * 3 * math.pi)
        base_x = 0.35 + 0.25 * t + wander
        base_y = 0.40 + 0.15 * t

        landmarks = []
        for i in range(21):
            # posições determinísticas e distintas por landmark, com um
            # movimentinho de dedo dependente do tempo
            landmarks.append(
                [
                    base_x + 0.012 * i + 0.01 * math.sin(t * math.pi + i),
                    base_y + 0.009 * ((i * 7) % 11) + 0.01 * math.cos(t * math.pi + i),
                    0.004 * ((i * 5) % 9) - 0.02 * t,
                ]
            )
        frames.append(landmarks)
        times.append(f * 40.0)  # ~25 fps
    return frames, times


def main():
    frames, times = synthetic_gesture()
    features = extract_features(frames, times)

    assert len(features) == N_FEATURES, (
        f"extract_features devolveu {len(features)} números, esperado {N_FEATURES}"
    )

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump({"frames": frames, "timesMs": times, "expected": features}, f)

    print(f"{len(features)} características calculadas (esperado: {N_FEATURES}).")
    print("Primeiras 8:", [round(v, 6) for v in features[:8]])
    print("Últimas 3 (retilineidade, inversões, duração):", [round(v, 6) for v in features[-3:]])
    print(f"\nFixture salvo em: {os.path.abspath(FIXTURE_PATH)}")


if __name__ == "__main__":
    main()
