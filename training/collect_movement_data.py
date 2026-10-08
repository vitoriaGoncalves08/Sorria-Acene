"""Grava TODAS as letras do alfabeto num só lugar: as com movimento viram
gestos (sequências de frames), as paradas viram poses.

Uso:
    python collect_movement_data.py

Controles:
    - H, J, K, X ou Z : conta e grava o gesto inteiro (RECORD_SECONDS).
      Faça o movimento completo da letra durante a gravação.
      -> data/movement_sequences.csv
    - Qualquer outra letra (A-Z): conta e grava uma rajada de poses
      (SAMPLES_PER_BURST amostras em BURST_SECONDS): mantenha a pose e mexa a
      mão levemente. É o mesmo que collect_data.py faz, com a mesma
      normalização (importada de lá) — pode usar qualquer um dos dois.
      -> data/landmarks.csv
    - BARRA DE ESPAÇO : grava uma amostra de "NADA" — mexa a mão à toa, leve
      ela até uma posição qualquer, coce o nariz. São essas amostras que
      ensinam o modelo a NÃO confundir "levei a mão até a posição" com letra.
    - ESC : sair.

Grave ~40 gestos de cada letra com movimento e ~60 de NADA, variando
velocidade, ângulo e distância da câmera. Para as paradas, ~30-50 poses.

⚠️ ANTES DE GRAVAR: confira numa fonte confiável (dicionário do INES ou vídeo
de sinalizante nativo) como cada uma dessas 5 letras é feita de verdade. Se
gravar o movimento errado, o modelo vai aprender o movimento errado com muita
precisão — o que é pior do que não ter a letra.

Saída: training/data/movement_sequences.csv, com os landmarks CRUS de cada
frame (sem normalizar). Guardamos crus de propósito: as características do
gesto são derivadas depois, em movement_features.py, então dá pra melhorar a
extração de características sem precisar regravar nada.
"""

import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Poses (letras paradas): mesma normalização e mesmas constantes do script que
# já grava poses, em vez de uma cópia que poderia sair de sincronia com ele.
from collect_data import BURST_SECONDS, OUTPUT_CSV as POSES_CSV, SAMPLES_PER_BURST, normalize

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "data", "movement_sequences.csv")
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "extension", "lib", "models", "hand_landmarker.task"
)

MOVEMENT_LETTERS = {"H", "J", "K", "X", "Z"}
NONE_LABEL = "NADA"

RECORD_SECONDS = 1.5
COUNTDOWN_SECONDS = 1.2  # tempo para se preparar depois de apertar a tecla

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 120, 255), -1)


def draw_banner(frame, text, color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Veja o README na raiz do projeto."
        )

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Não consegui abrir a webcam. Feche a aba da extensão no Chrome e "
            "outros apps de vídeo, e confira em Configurações do Windows > "
            "Privacidade e segurança > Câmera se apps de área de trabalho "
            "podem acessar a câmera."
        )

    poses_exist = os.path.exists(POSES_CSV)
    os.makedirs(os.path.dirname(POSES_CSV), exist_ok=True)

    counts = {}
    state = "idle"  # idle -> countdown -> recording (gesto) | burst (pose)
    kind = "gesture"  # o que está sendo gravado: "gesture" | "pose"
    label = None
    state_started_at = 0.0
    recorded = []  # [(t_ms, landmarks)] do gesto em andamento
    burst_count = 0
    last_burst_at = 0.0
    burst_interval = BURST_SECONDS / SAMPLES_PER_BURST

    with open(OUTPUT_CSV, "a", newline="") as f, open(
        POSES_CSV, "a", newline=""
    ) as fp, vision.HandLandmarker.create_from_options(options) as landmarker:
        writer = csv.writer(f)
        pose_writer = csv.writer(fp)
        if not file_exists:
            header = ["sequence_id", "frame", "t_ms"]
            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]
            header.append("label")
            writer.writerow(header)
        if not poses_exist:
            pose_writer.writerow([f"p{i}" for i in range(63)] + ["label"])

        print("Qualquer letra A-Z = gravar | ESPAÇO = gravar 'NADA' | ESC = sair")
        print("  H J K X Z gravam o gesto; as outras gravam uma rajada de poses.")
        clock_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now = time.time()
            timestamp_ms = int((now - clock_start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand:
                draw_landmarks(frame, hand)

            if state == "countdown":
                remaining = COUNTDOWN_SECONDS - (now - state_started_at)
                if remaining <= 0:
                    state = "recording" if kind == "gesture" else "burst"
                    state_started_at = now
                    recorded = []
                    burst_count = 0
                    last_burst_at = 0.0
                else:
                    draw_banner(frame, f"'{label}' em {remaining:.1f}s — prepare-se", (0, 255, 255))

            if state == "burst":
                # rajada de poses de uma letra parada: mesmo critério do collect_data.py
                if hand and (now - last_burst_at) >= burst_interval:
                    pose_writer.writerow(list(normalize(hand)) + [label])
                    burst_count += 1
                    last_burst_at = now
                draw_banner(
                    frame,
                    f"GRAVANDO '{label}': {burst_count}/{SAMPLES_PER_BURST} — mexa a mão um pouco",
                    (0, 0, 255),
                )
                # o limite de tempo evita travar se a mão sair do quadro
                if burst_count >= SAMPLES_PER_BURST or (now - state_started_at) > BURST_SECONDS * 3:
                    fp.flush()
                    counts[label] = counts.get(label, 0) + burst_count
                    print(f"'{label}' (pose): {burst_count} amostras — total nesta sessão: {counts[label]}")
                    state = "idle"
                    label = None

            if state == "recording":
                elapsed = now - state_started_at
                if hand:
                    recorded.append((timestamp_ms, hand))
                draw_banner(
                    frame,
                    f"GRAVANDO '{label}' — faça o movimento ({elapsed:.1f}/{RECORD_SECONDS:.1f}s)",
                    (0, 0, 255),
                )
                if elapsed >= RECORD_SECONDS:
                    if len(recorded) >= 5:
                        seq_id = int(now * 1000)
                        for frame_index, (t_ms, lm) in enumerate(recorded):
                            row = [seq_id, frame_index, t_ms]
                            for p in lm:
                                row += [p.x, p.y, p.z]
                            row.append(label)
                            writer.writerow(row)
                        f.flush()
                        counts[label] = counts.get(label, 0) + 1
                        print(f"'{label}' gravado ({len(recorded)} frames) — total: {counts[label]}")
                    else:
                        print("Mão detectada em poucos frames, gesto descartado.")
                    state = "idle"
                    label = None

            if state == "idle":
                draw_banner(frame, "A-Z = gravar letra | ESPACO = NADA | ESC = sair", (255, 255, 255))

            cv2.imshow("Coleta de letras - poses e gestos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if state == "idle":
                if key == 32:  # espaço
                    label, kind = NONE_LABEL, "gesture"
                    state = "countdown"
                    state_started_at = now
                elif 65 <= key <= 90 or 97 <= key <= 122:
                    label = chr(key).upper()
                    kind = "gesture" if label in MOVEMENT_LETTERS else "pose"
                    state = "countdown"
                    state_started_at = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
