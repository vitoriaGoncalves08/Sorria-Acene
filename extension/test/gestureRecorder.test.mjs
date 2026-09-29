// Testa a máquina de estados que separa "mão parada" (pose) de "mão fazendo
// um gesto" — a peça que decide qual dos dois classificadores usar.
//
// Não precisa de webcam: simula sequências de mãos com posições controladas.
//
// Uso: cd extension && npm test
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { createGestureRecorder } = await import(
  pathToFileURL(path.join(here, "..", "src", "app", "gestureRecorder.js")).href
);

const FPS_MS = 33; // ~30 quadros por segundo
const HAND_SIZE = 0.1; // distância pulso -> landmark 9 nas mãos simuladas

// Uma "mão" de 21 pontos com o pulso em (x, y). Só o pulso e o landmark 9
// importam aqui: são os que o gestureRecorder usa (posição e régua da mão).
function hand(x, y = 0.5) {
  const pts = [];
  for (let i = 0; i < 21; i++) pts.push({ x, y, z: 0 });
  pts[9] = { x, y: y + HAND_SIZE, z: 0 };
  return pts;
}

// Deslocamento por quadro para uma dada velocidade em tamanhos-de-mão/segundo
const perFrame = (speed) => speed * HAND_SIZE * (FPS_MS / 1000);

function still(n, pos) {
  return { steps: Array.from({ length: n }, () => hand(pos)), pos };
}

function move(n, speed, pos) {
  const steps = [];
  for (let i = 0; i < n; i++) {
    pos += perFrame(speed);
    steps.push(hand(pos));
  }
  return { steps, pos };
}

function gone(n, pos) {
  return { steps: Array.from({ length: n }, () => null), pos };
}

// Encadeia trechos mantendo a posição contínua — se a mão "teletransportasse"
// entre trechos, isso viraria um pico de velocidade artificial.
function sequence(...parts) {
  let pos = 0.3;
  const steps = [];
  for (const [fn, ...args] of parts) {
    const result = fn(...args, pos);
    steps.push(...result.steps);
    pos = result.pos;
  }
  return steps;
}

function countGestures(steps) {
  const fired = [];
  const recorder = createGestureRecorder({
    onGesture: (frames, times) =>
      fired.push({ frames: frames.length, durMs: times[times.length - 1] - times[0] }),
  });
  let t = 0;
  for (const landmarks of steps) {
    recorder.update(landmarks, t);
    t += FPS_MS;
  }
  return fired;
}

const cases = [
  ["mão totalmente parada", sequence([still, 60]), 0],
  ["gesto normal (~1s de movimento)", sequence([still, 15], [move, 30, 6.0], [still, 25]), 1],
  ["tremida curta de 3 quadros", sequence([still, 15], [move, 3, 6.0], [still, 25]), 0],
  ["movimento lento, abaixo do limiar", sequence([still, 15], [move, 30, 0.5], [still, 25]), 0],
  ["mão some do quadro no meio do gesto", sequence([still, 10], [move, 20, 6.0], [gone, 10]), 0],
  [
    "dois gestos seguidos",
    sequence([still, 10], [move, 25, 6.0], [still, 20], [move, 25, 6.0], [still, 20]),
    2,
  ],
];

let failures = 0;
for (const [name, steps, expected] of cases) {
  const fired = countGestures(steps);
  const ok = fired.length === expected;
  if (!ok) failures++;
  const detail = fired.length ? `  ${JSON.stringify(fired)}` : "";
  console.log(`${ok ? "OK   " : "FALHA"} ${name}: ${fired.length}/${expected}${detail}`);
}

if (failures > 0) {
  console.error(`\n${failures} caso(s) falharam.`);
  process.exit(1);
}
console.log("\nTodos os casos passaram.");
