// Classifica um GESTO (sequência de frames) em uma das letras com movimento
// do alfabeto de Libras: H, J, K, X, Z.
//
// Essas 5 letras não são poses — a mão se desloca ou gira. Por isso o
// classificador estático (classifier.js), que julga uma "foto" por vez, não
// dá conta delas: no meio de um J a mão passa por formas que parecem outras
// letras paradas.
//
// Existe uma sexta classe, "NADA", treinada com gravações de mão se mexendo
// à toa. É ela que evita que simplesmente levar a mão até a posição seja
// interpretado como letra — em vez de eu tentar adivinhar por regra o que é
// gesto e o que é transição, o próprio modelo decide.
//
// Contrato:
//   await loadModel()   -> prepara (no-op se o modelo ainda não foi treinado)
//   isReady() -> boolean
//   predictGesture(frames, timesMs) -> { letter, confidence } | null
//     letter é "NADA" quando o movimento não foi letra nenhuma
import { forwardPass, loadModelFile } from "./mlp.js";
import { extractFeatures } from "./movementFeatures.js";

export const NONE_LABEL = "NADA";

let model = null;

export async function loadModel() {
  model = await loadModelFile("lib/models/movement_classifier.json");
  return model;
}

export function isReady() {
  return model !== null;
}

// Só as letras — a classe NADA é mecanismo interno, não faz parte do alfabeto.
export function getLabels() {
  return model ? model.labels.filter((l) => l !== NONE_LABEL) : [];
}

export function predictGesture(frames, timesMs) {
  if (!isReady()) return null;

  const features = extractFeatures(frames, timesMs);
  if (!features) return null;

  const { label, confidence } = forwardPass(model, features);
  return { letter: label, confidence };
}
