// Extrai as características de um GESTO (sequência de frames) para o
// classificador das letras com movimento (H, J, K, X, Z).
//
// ⚠️ ESTE ARQUIVO TEM UM GÊMEO EM PYTHON ⚠️
// `training/movement_features.py` precisa calcular EXATAMENTE os mesmos
// números, na mesma ordem. Se mexer aqui, mexa lá também — senão o modelo
// recebe em tempo real entradas diferentes das que viu no treino, e passa a
// errar sem dar nenhum erro visível.
//
// Ordem das 60 características (ver o gêmeo em Python para o porquê de cada):
//   forma da mão no início   21
//   forma da mão no fim      21
//   trajeto do pulso         15
//   retilineidade             1
//   inversões de direção      1
//   duração                   1

const KEY_POINTS = [4, 8, 12, 16, 20, 9, 5];
const N_KEYFRAMES = 6;
const WRIST = 0;
const MIDDLE_MCP = 9;
const DEADZONE_RATIO = 0.02;
const MAX_STRAIGHTNESS = 20.0;

export const N_FEATURES = KEY_POINTS.length * 3 * 2 + (N_KEYFRAMES - 1) * 3 + 3;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function handSize(frames) {
  let sum = 0;
  for (const f of frames) sum += dist(f[MIDDLE_MCP], f[WRIST]);
  return Math.max(sum / frames.length, 1e-6);
}

function normalizedKeyPoints(frame) {
  const wrist = frame[WRIST];
  const centered = frame.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  // maior distância medida sobre TODOS os 21 pontos (igual ao Python), só
  // depois selecionamos os 7 de KEY_POINTS
  let maxDist = 1e-6;
  for (const p of centered) {
    const d = Math.hypot(p.x, p.y, p.z);
    if (d > maxDist) maxDist = d;
  }
  const out = [];
  for (const i of KEY_POINTS) {
    out.push(centered[i].x / maxDist, centered[i].y / maxDist, centered[i].z / maxDist);
  }
  return out;
}

function keyframeIndices(nFrames) {
  if (nFrames === 1) return new Array(N_KEYFRAMES).fill(0);
  const idx = [];
  for (let i = 0; i < N_KEYFRAMES; i++) {
    idx.push(Math.round((i * (nFrames - 1)) / (N_KEYFRAMES - 1)));
  }
  return idx;
}

function countReversals(wristTrack, size) {
  const first = wristTrack[0];
  const last = wristTrack[wristTrack.length - 1];
  const axis = Math.abs(last.x - first.x) >= Math.abs(last.y - first.y) ? "x" : "y";
  const deadzone = DEADZONE_RATIO * size;

  let reversals = 0;
  let lastSign = 0;
  for (let i = 1; i < wristTrack.length; i++) {
    const delta = wristTrack[i][axis] - wristTrack[i - 1][axis];
    if (Math.abs(delta) < deadzone) continue;
    const sign = delta > 0 ? 1 : -1;
    if (lastSign !== 0 && sign !== lastSign) reversals++;
    lastSign = sign;
  }
  return reversals;
}

// frames: array de frames, cada um com os 21 landmarks CRUS {x,y,z} do
// MediaPipe (sem normalizar — o trajeto depende de onde a mão está no quadro).
// timesMs: array de timestamps, do mesmo tamanho.
export function extractFeatures(frames, timesMs) {
  if (frames.length < 2) return null;

  const size = handSize(frames);
  const keyIdx = keyframeIndices(frames.length);
  const wristTrack = frames.map((f) => f[WRIST]);

  const features = [];
  features.push(...normalizedKeyPoints(frames[keyIdx[0]]));
  features.push(...normalizedKeyPoints(frames[keyIdx[keyIdx.length - 1]]));

  const origin = wristTrack[keyIdx[0]];
  for (let k = 1; k < keyIdx.length; k++) {
    const w = wristTrack[keyIdx[k]];
    features.push((w.x - origin.x) / size, (w.y - origin.y) / size, (w.z - origin.z) / size);
  }

  // Escalares sobre TODOS os frames, não só os keyframes — senão o
  // zigue-zague do Z passaria despercebido entre um keyframe e outro.
  let pathLength = 0;
  for (let i = 1; i < wristTrack.length; i++) {
    pathLength += dist(wristTrack[i], wristTrack[i - 1]);
  }
  pathLength /= size;
  const netDist = dist(wristTrack[wristTrack.length - 1], wristTrack[0]) / size;
  const straightness = Math.min(pathLength / Math.max(netDist, 1e-6), MAX_STRAIGHTNESS);
  const reversals = countReversals(wristTrack, size);
  const duration = (timesMs[timesMs.length - 1] - timesMs[0]) / 1000;

  features.push(straightness, reversals, duration);
  return features;
}
