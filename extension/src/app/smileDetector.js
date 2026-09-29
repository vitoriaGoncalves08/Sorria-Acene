// Detecta um sorriso a partir dos blendshapes do rosto (MediaPipe
// FaceLandmarker com `outputFaceBlendshapes: true`) — mais simples e mais
// confiável que detectar aceno de cabeça: não exige nenhuma conta de
// geometria (o MediaPipe já entrega um número de 0 a 1 pra "quanto está
// sorrindo"), e sorrir não mexe a mão que está fazendo a letra, então não
// atrapalha o reconhecimento como o aceno atrapalhava.
//
// Precisa segurar o sorriso por um tempinho curto (HOLD_MS) pra confirmar —
// evita disparar com uma risadinha rápida sem querer.
const SMILE_THRESHOLD = 0.5; // acima disso conta como "sorrindo"
const RESET_THRESHOLD = 0.3; // precisa cair abaixo disso pra poder confirmar de novo
const HOLD_MS = 350; // precisa manter o sorriso por esse tempo pra confirmar
const COOLDOWN_MS = 1200; // tempo ignorando novos sorrisos depois de confirmar um

function getSmileScore(blendshapes) {
  if (!blendshapes || !blendshapes.categories) return null;
  const left = blendshapes.categories.find((c) => c.categoryName === "mouthSmileLeft");
  const right = blendshapes.categories.find((c) => c.categoryName === "mouthSmileRight");
  if (!left || !right) return null;
  return (left.score + right.score) / 2;
}

export function createSmileDetector({ onSmile } = {}) {
  let smileStartT = null;
  let armed = true; // false logo após confirmar, até o sorriso "resetar"
  let cooldownUntil = 0;

  function update(blendshapes, now = performance.now()) {
    const score = getSmileScore(blendshapes);
    if (score === null) return false;

    if (now < cooldownUntil) return false;

    if (!armed) {
      if (score < RESET_THRESHOLD) armed = true;
      return false;
    }

    if (score >= SMILE_THRESHOLD) {
      if (smileStartT === null) smileStartT = now;
      if (now - smileStartT >= HOLD_MS) {
        armed = false;
        cooldownUntil = now + COOLDOWN_MS;
        smileStartT = null;
        if (onSmile) onSmile();
        return true;
      }
    } else {
      smileStartT = null;
    }
    return false;
  }

  return { update };
}
