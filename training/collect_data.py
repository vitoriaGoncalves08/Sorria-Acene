"""Coleta amostras de landmarks da mão para treinar o classificador de letras.

Uso:
    python collect_data.py

Controles:
    - Pressione uma tecla A-Z: inicia uma gravação automática de várias
      amostras daquela letra (SAMPLES_PER_BURST, ao longo de BURST_SECONDS).
      Mantenha a pose e mexa levemente a mão (ângulo, distância da câmera)
      enquanto ele grava — isso ensina o modelo a reconhecer a letra em
      variações, não só numa posição exata.
    - Pressione ESC para sair.

Cada amostra gravada é uma linha no CSV de saída com 63 números (21 pontos *
x,y,z, já normalizados/centralizados no pulso — mesma normalização usada em
`extension/src/classifier/classifier.js`, para o modelo treinado bater com o
que a extensão calcula em tempo real) mais a coluna `label` com a letra.

Usa a mesma API (MediaPipe Tasks) e o mesmo modelo (hand_landmarker.task) que
a extensão usa em JavaScript, pra garantir que os pontos detectados aqui
sejam consistentes com os que a extensão vai detectar depois.
"""

import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "extension", "lib", "models", "hand_landmarker.task"
)

SAMPLES_PER_BURST = 30
BURST_SECONDS = 3.0
COUNTDOWN_SECONDS = 0.6  # pausa curta após apertar a tecla, antes de começar a gravar

# Mesmas conexões desenhadas em extension/src/app/app.js, só para dar
# feedback visual de que a mão está sendo bem detectada.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    max_dist = max(np.linalg.norm(centered, axis=1).max(), 1e-6)
    return (centered / max_dist).flatten()


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
            f"Modelo não encontrado em {MODEL_PATH}. Rode a preparação da "
            "extensão primeiro (veja o README na raiz do projeto) ou baixe "
            "manualmente hand_landmarker.task para essa pasta."
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
            "Não consegui abrir a webcam (índice 0). Possíveis causas: outro "
            "programa está usando a câmera (feche a aba da extensão no Chrome "
            "e outros apps de vídeo), ou o Windows está bloqueando apps de "
            "área de trabalho — confira em Configurações do Windows > "
            "Privacidade e segurança > Câmera > 'Permitir que aplicativos de "
            "área de trabalho acessem sua câmera'."
        )
    counts = {}

    # Estado da gravação em rajada: "idle" | "countdown" | "capturing"
    state = "idle"
    burst_letter = None
    burst_count = 0
    state_started_at = 0.0
    last_capture_at = 0.0
    capture_interval = BURST_SECONDS / SAMPLES_PER_BURST

    with open(OUTPUT_CSV, "a", newline="") as f, vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([f"p{i}" for i in range(63)] + ["label"])

        print(
            f"Pressione A-Z: grava {SAMPLES_PER_BURST} amostras automaticamente "
            f"em {BURST_SECONDS:.0f}s (mexa a mão levemente enquanto grava). "
            "ESC para sair."
        )
        clock_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - clock_start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand_landmarks:
                draw_landmarks(frame, hand_landmarks)

            now = time.time()

            if state == "countdown":
                if now - state_started_at >= COUNTDOWN_SECONDS:
                    state = "capturing"
                    last_capture_at = 0.0
                else:
                    draw_banner(frame, f"Prepare a pose de '{burst_letter}'...", (0, 255, 255))

            if state == "capturing":
                remaining = SAMPLES_PER_BURST - burst_count
                draw_banner(
                    frame,
                    f"Gravando '{burst_letter}': {burst_count}/{SAMPLES_PER_BURST} "
                    "— mexa a mão um pouco",
                    (0, 255, 0),
                )
                if hand_landmarks and (now - last_capture_at) >= capture_interval:
                    features = normalize(hand_landmarks)
                    writer.writerow(list(features) + [burst_letter])
                    burst_count += 1
                    last_capture_at = now
                    counts[burst_letter] = counts.get(burst_letter, 0) + 1
                if remaining <= 0:
                    f.flush()
                    print(f"'{burst_letter}' completo (total acumulado: {counts[burst_letter]})")
                    state = "idle"
                    burst_letter = None

            if state == "idle":
                draw_banner(
                    frame,
                    "Pressione A-Z para gravar uma letra (ESC para sair)",
                    (255, 255, 255),
                )

            cv2.imshow("Coleta de dados - Alfabeto Libras", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if state == "idle" and (65 <= key <= 90 or 97 <= key <= 122):  # A-Z ou a-z
                burst_letter = chr(key).upper()
                burst_count = 0
                state_started_at = now
                state = "countdown"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
