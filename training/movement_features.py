"""Extrai as características de um GESTO (sequência de frames) para o
classificador das letras com movimento (H, J, K, X, Z).

⚠️ ESTE ARQUIVO TEM UM GÊMEO EM JAVASCRIPT ⚠️
`extension/src/classifier/movementFeatures.js` precisa calcular EXATAMENTE os
mesmos números, na mesma ordem. Se você mexer aqui, mexa lá também — senão o
modelo treinado recebe entradas diferentes das que viu no treino, e passa a
errar sem dar nenhum erro visível.

Por que características e não a sequência crua: um gesto pode durar 0,8s ou
1,5s, e o classificador precisa de entrada de tamanho fixo. Então reamostramos
a sequência em N_KEYFRAMES instantes e resumimos o gesto em 60 números:

  forma da mão no início   21  (7 pontos * xyz, normalizados no pulso)
  forma da mão no fim      21  ← pega o J, que gira o pulso
  trajeto do pulso         15  (5 pontos, relativos ao início)
  retilineidade             1  ← separa K (sobe reto) de Z (zigue-zague)
  inversões de direção      1  ← Z tem ~2, K tem 0
  duração                   1
                          ----
                           60
"""

import numpy as np

# Pontas dos dedos + duas juntas da palma. Não incluímos o pulso (índice 0)
# porque, depois de centralizar nele, ele é sempre (0,0,0) — não carrega
# informação nenhuma.
KEY_POINTS = [4, 8, 12, 16, 20, 9, 5]

N_KEYFRAMES = 6
WRIST = 0
MIDDLE_MCP = 9  # junta da base do dedo médio, usada como "régua" da mão
DEADZONE_RATIO = 0.02  # movimento menor que isso * tamanho da mão é tremor, não gesto
MAX_STRAIGHTNESS = 20.0  # teto, pra não explodir quando o gesto volta ao ponto de partida

N_FEATURES = len(KEY_POINTS) * 3 * 2 + (N_KEYFRAMES - 1) * 3 + 3


def _hand_size(frames):
    """Distância pulso -> base do dedo médio, média de todos os frames.

    Serve de régua anatômica: deixa o trajeto invariante à distância da
    pessoa até a câmera (mão longe = deslocamento menor em pixels).
    """
    sizes = [
        np.linalg.norm(np.array(f[MIDDLE_MCP]) - np.array(f[WRIST])) for f in frames
    ]
    return max(float(np.mean(sizes)), 1e-6)


def _normalized_key_points(frame):
    """Forma da mão num frame: mesma normalização do classificador estático
    (centraliza no pulso, escala pela maior distância), mas devolvendo só os
    7 pontos de KEY_POINTS em vez dos 21 — o gesto não precisa da mão inteira,
    e menos números significa menos risco de o modelo decorar em vez de
    aprender.
    """
    pts = np.array(frame, dtype=np.float64)
    centered = pts - pts[WRIST]
    max_dist = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return (centered[KEY_POINTS] / max_dist).flatten()


def _keyframe_indices(n_frames):
    """N_KEYFRAMES índices igualmente espaçados, incluindo o primeiro e o
    último frame."""
    if n_frames == 1:
        return [0] * N_KEYFRAMES
    return [
        int(round(i * (n_frames - 1) / (N_KEYFRAMES - 1))) for i in range(N_KEYFRAMES)
    ]


RESAMPLE_MS = 1000.0 / 30.0  # taxa fixa do trajeto: 30 quadros por segundo


def _resample_track(track, times_ms):
    """Trajeto do pulso interpolado numa grade fixa de RESAMPLE_MS, para que
    as características não dependam da taxa de quadros de quem gravou."""
    t = np.asarray(times_ms, dtype=np.float64)
    n = max(2, int(np.floor((t[-1] - t[0]) / RESAMPLE_MS)) + 1)
    grid = np.minimum(t[0] + np.arange(n) * RESAMPLE_MS, t[-1])
    return np.stack([np.interp(grid, t, track[:, c]) for c in range(3)], axis=1)


def _count_reversals(wrist_track, hand_size):
    """Quantas vezes a mão inverteu o sentido ao longo do eixo em que mais
    andou. É o que distingue o zigue-zague do Z do movimento reto do K."""
    total = wrist_track[-1] - wrist_track[0]
    axis = 0 if abs(total[0]) >= abs(total[1]) else 1  # x ou y (z é ruidoso demais)
    deadzone = DEADZONE_RATIO * hand_size

    reversals = 0
    last_sign = 0
    for i in range(1, len(wrist_track)):
        delta = wrist_track[i][axis] - wrist_track[i - 1][axis]
        if abs(delta) < deadzone:
            continue
        sign = 1 if delta > 0 else -1
        if last_sign != 0 and sign != last_sign:
            reversals += 1
        last_sign = sign
    return float(reversals)


def extract_features(frames, times_ms):
    """frames: lista de frames; cada frame é uma lista de 21 pontos [x, y, z]
    CRUS (como o MediaPipe entrega, sem normalizar) — precisamos dos valores
    crus porque o trajeto depende de onde a mão está no quadro.
    times_ms: lista de timestamps em milissegundos, do mesmo tamanho.

    Devolve uma lista de N_FEATURES números.
    """
    if len(frames) < 2:
        raise ValueError("Um gesto precisa de pelo menos 2 frames.")

    hand_size = _hand_size(frames)
    key_idx = _keyframe_indices(len(frames))
    wrist_track = np.array([f[WRIST] for f in frames], dtype=np.float64)

    shape_start = _normalized_key_points(frames[key_idx[0]])
    shape_end = _normalized_key_points(frames[key_idx[-1]])

    # Trajeto: onde o pulso estava em cada keyframe, relativo ao ponto de
    # partida. Pulamos o primeiro keyframe porque ele é sempre (0,0,0).
    origin = wrist_track[key_idx[0]]
    trajectory = []
    for i in key_idx[1:]:
        trajectory.extend((wrist_track[i] - origin) / hand_size)

    # Escalares calculados sobre TODOS os frames (não só os keyframes), senão
    # o zigue-zague do Z passaria despercebido entre um keyframe e outro — mas
    # sobre o trajeto REAMOSTRADO numa taxa fixa. Sem isso, a contagem de
    # inversões depende de quantos quadros por segundo a câmera/navegador
    # entrega: os gestos gravados pelo script (~30 fps) tinham ~2 inversões, e
    # os mesmos gestos feitos na extensão (~60 fps) chegavam a 7, porque cada
    # tremidinha da mão virava uma inversão. O modelo era treinado com um e
    # recebia o outro.
    track = _resample_track(wrist_track, times_ms)
    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    path_length = float(steps.sum()) / hand_size
    net_dist = float(np.linalg.norm(track[-1] - track[0])) / hand_size
    straightness = min(path_length / max(net_dist, 1e-6), MAX_STRAIGHTNESS)
    reversals = _count_reversals(track, hand_size)
    duration = (times_ms[-1] - times_ms[0]) / 1000.0

    return (
        list(shape_start)
        + list(shape_end)
        + list(trajectory)
        + [straightness, reversals, duration]
    )
