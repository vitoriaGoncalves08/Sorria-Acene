"""Processa o Brazilian Sign Language Alphabet Dataset (imagens) e extrai os
landmarks da mão de cada imagem com o MesmoLandmarker (MediaPipe Tasks) usado
no resto do projeto — gerando um CSV no MESMO formato de
training/data/landmarks.csv, mas em ARQUIVO SEPARADO, sem misturar com as
suas próprias amostras (por pedido explícito).

Como conseguir o dataset:
  1. Baixe o zip em https://data.mendeley.com/datasets/k4gs3bmx5k/5
     (ou o link direto no README do repositório
     github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset)
  2. Extraia de forma que fique:
       training/data/external/extracted/A/*.jpg (+ .xml)
       training/data/external/extracted/B/*.jpg (+ .xml)
       ...

Só cobre 15 letras estáticas (A, B, C, D, E, I, L, M, N, O, R, S, U, V, W) —
o próprio dataset não inclui as letras com movimento (H, J, K, X, Z), que não
têm como ser representadas numa imagem parada.

IMPORTANTE (ler antes de usar pra valer): essas imagens são originalmente de
um dataset de ASL (língua de sinais americana) — os autores selecionaram as
15 letras em que afirmam que a forma da mão é a mesma em Libras. É um
trabalho acadêmico publicado, mas não foi capturado com sinalizantes
brasileiros nativos. Vale conferir visualmente algumas amostras antes de
confiar cegamente (ver função `--preview` abaixo).

Uso:
    python process_external_dataset.py
    python process_external_dataset.py --preview   # só mostra 1 imagem por letra, não processa tudo
"""

import argparse
import os
import xml.etree.ElementTree as ET

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "external", "extracted")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "data", "external", "landmarks_external.csv")
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "extension", "lib", "models", "hand_landmarker.task"
)


def normalize(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    max_dist = max(np.linalg.norm(centered, axis=1).max(), 1e-6)
    return (centered / max_dist).flatten()


def read_bbox(xml_path):
    if not os.path.exists(xml_path):
        return None
    root = ET.parse(xml_path).getroot()
    box = root.find(".//bndbox")
    if box is None:
        return None
    return (
        int(box.find("xmin").text),
        int(box.find("ymin").text),
        int(box.find("xmax").text),
        int(box.find("ymax").text),
    )


def cropped_with_padding(image, bbox, padding_ratio=0.4):
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    box_w, box_h = xmax - xmin, ymax - ymin
    pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)
    x0 = max(0, xmin - pad_x)
    y0 = max(0, ymin - pad_y)
    x1 = min(w, xmax + pad_x)
    y1 = min(h, ymax + pad_y)
    crop = image[y0:y1, x0:x1]
    # imagens pequenas prejudicam a detecção; aumenta se necessário
    if crop.shape[0] < 256:
        scale = 256 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def detect(landmarker, image_bgr):
    # Espelha antes de detectar, na mesma convenção de collect_data.py (que
    # espelha o quadro da webcam). Sem isso, as amostras do dataset externo
    # saíam como a "outra mão" em relação às suas, e o modelo era treinado com
    # as duas orientações misturadas.
    image_bgr = cv2.flip(image_bgr, 1)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    return result.hand_landmarks[0] if result.hand_landmarks else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="mostra 1 imagem por letra, sem processar tudo")
    args = parser.parse_args()

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATASET_DIR}. Baixe e extraia primeiro "
            "(veja as instruções no topo deste arquivo)."
        )

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    letters = sorted(
        d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        if args.preview:
            for letter in letters:
                folder = os.path.join(DATASET_DIR, letter)
                first_jpg = next(f for f in sorted(os.listdir(folder)) if f.endswith(".jpg"))
                img = cv2.imread(os.path.join(folder, first_jpg))
                landmarks = detect(landmarker, img)
                status = "mão detectada" if landmarks else "SEM detecção"
                cv2.putText(img, f"{letter}: {status}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow(f"Preview {letter} (qualquer tecla p/ proxima, ESC p/ sair)", img)
                if cv2.waitKey(0) & 0xFF == 27:
                    break
                cv2.destroyAllWindows()
            return

        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        rows_written = 0
        stats = {}

        with open(OUTPUT_CSV, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow([f"p{i}" for i in range(63)] + ["label"])

            for letter in letters:
                folder = os.path.join(DATASET_DIR, letter)
                jpgs = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
                ok_count = 0

                for jpg in jpgs:
                    img_path = os.path.join(folder, jpg)
                    xml_path = os.path.join(folder, os.path.splitext(jpg)[0] + ".xml")
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    landmarks = detect(landmarker, img)

                    if landmarks is None:
                        bbox = read_bbox(xml_path)
                        if bbox is not None:
                            crop = cropped_with_padding(img, bbox)
                            landmarks = detect(landmarker, crop)

                    if landmarks is not None:
                        features = normalize(landmarks)
                        writer.writerow(list(features) + [letter])
                        rows_written += 1
                        ok_count += 1

                stats[letter] = (ok_count, len(jpgs))
                print(f"{letter}: {ok_count}/{len(jpgs)} imagens com mão detectada")

        print(f"\nTotal de amostras gravadas: {rows_written}")
        print(f"Arquivo: {os.path.abspath(OUTPUT_CSV)}")
        print("(separado de training/data/landmarks.csv, não foi misturado)")


if __name__ == "__main__":
    main()
