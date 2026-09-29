// Classifica uma pose PARADA de mão (21 landmarks do MediaPipe) em uma letra
// do alfabeto. Para as letras com movimento (H, J, K, X, Z), ver
// movementClassifier.js — elas não cabem numa pose única.
//
// O modelo é treinado fora da extensão em training/train.py (scikit-learn) e
// exportado como JSON puro (pesos de uma rede neural pequena) em
// lib/models/letter_classifier.json — ver training/README.md. Não usamos
// TensorFlow.js aqui de propósito: o modelo é pequeno o bastante pra rodar
// com um forward pass manual, sem precisar carregar uma biblioteca de ML
// inteira no navegador.
//
// Contrato:
//   await loadModel()               -> prepara o classificador (no-op se o
//                                       arquivo do modelo ainda não existir)
//   isReady() -> boolean
//   predict(landmarks) -> { letter: string, confidence: number } | null
//     landmarks: array de 21 pontos {x, y, z} normalizados (saída do HandLandmarker)
import { forwardPass, loadModelFile } from "./mlp.js";

let model = null; // { labels: string[], layers: [{weights, biases, activation}] }

export async function loadModel() {
  model = await loadModelFile("lib/models/letter_classifier.json");
  return model;
}

export function isReady() {
  return model !== null;
}

// Quais letras este modelo conhece — a tela usa isso pra montar o alfabeto,
// então ela nunca fica desatualizada em relação ao que foi treinado.
export function getLabels() {
  return model ? model.labels : [];
}

export function normalizeLandmarks(landmarks) {
  // Centraliza no pulso (ponto 0) e normaliza pela maior distância, para o
  // classificador ser invariante a posição/tamanho da mão no quadro. Precisa
  // ser EXATAMENTE a mesma conta feita em training/collect_data.py, senão o
  // modelo recebe entradas diferentes das que foi treinado.
  const wrist = landmarks[0];
  const centered = landmarks.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  const maxDist = Math.max(
    ...centered.map((p) => Math.hypot(p.x, p.y, p.z)),
    1e-6
  );
  return centered.map((p) => [p.x / maxDist, p.y / maxDist, p.z / maxDist]).flat();
}

export function predict(landmarks) {
  if (!isReady()) return null;
  const { label, confidence } = forwardPass(model, normalizeLandmarks(landmarks));
  return { letter: label, confidence };
}
