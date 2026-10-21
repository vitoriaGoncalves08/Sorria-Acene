// A letra de um gesto reconhecido não pode ser trocada pela letra que o
// classificador de poses lê na pose em que a mão TERMINA o gesto (no Z, ele
// lê N e M), mas também não pode prender a tela para sempre.
//
// Uso: cd extension && npm test
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { createMovementHold, MIN_HOLD_MS, MAX_HOLD_MS } = await import(
  pathToFileURL(path.join(here, "..", "src", "app", "movementHold.js")).href
);

const THRESHOLD = 0.8;
const FRAME_MS = 33;

function gesture(hold) {
  hold.start(
    { letter: "Z", confidence: 0.97, frames: [], times: [], endLetters: new Set(["N", "M"]) },
    0
  );
}

// roda `seconds` de quadros com a mesma leitura estática e devolve quando o
// gesto saiu da tela (ms), ou null se ainda estava lá no fim
function runUntilReleased(hold, seconds, frame) {
  for (let now = 0; now <= seconds * 1000; now += FRAME_MS) {
    const active = hold.update({ now, ...frame(now) }, THRESHOLD);
    if (!active) return now;
  }
  return null;
}

const same = { hasHand: true, gestureInProgress: false, staticLetter: "N", staticConfidence: 0.99 };
const cases = [
  [
    "continua na tela enquanto a mão lê a letra do fim do gesto (N)",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, () => same);
    },
    (r) => r === null,
  ],
  [
    "também segura quando a leitura pisca entre N e M",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, (t) => ({ ...same, staticLetter: Math.floor(t / 100) % 2 ? "N" : "M" }));
    },
    (r) => r === null,
  ],
  [
    "solta quando você muda para uma pose de outra letra (depois do tempo mínimo)",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, (t) => (t < 3000 ? same : { ...same, staticLetter: "V" }));
    },
    (r) => r !== null && r >= 3000,
  ],
  [
    "nunca solta antes do tempo mínimo, mesmo mudando de pose",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, () => ({ ...same, staticLetter: "V" }));
    },
    (r) => r !== null && r >= MIN_HOLD_MS,
  ],
  [
    "um piscar de 1-2 quadros de outra letra não solta",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, (t) => (t > 2500 && t < 2560 ? { ...same, staticLetter: "V" } : same));
    },
    (r) => r === null,
  ],
  [
    "solta quando a mão sai do quadro",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, (t) => (t < 2500 ? same : { ...same, hasHand: false }));
    },
    (r) => r !== null && r >= 2500 && r < 2600,
  ],
  [
    "mão se mexendo de novo NÃO solta o gesto (ele pode ter sido cortado em dois pedaços)",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 6, (t) => (t < 2500 ? same : { ...same, gestureInProgress: true, staticLetter: "V" }));
    },
    (r) => r === null,
  ],
  [
    "depois de se mexer, parar numa pose de outra letra solta o gesto",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 8, (t) =>
        t < 2500 ? same : t < 4000 ? { ...same, gestureInProgress: true } : { ...same, staticLetter: "V" }
      );
    },
    (r) => r !== null && r >= 4000 && r < 4300,
  ],
  [
    "rede de segurança: solta depois do tempo máximo",
    () => {
      const h = createMovementHold();
      gesture(h);
      return runUntilReleased(h, 12, () => same);
    },
    (r) => r !== null && r >= MAX_HOLD_MS && r < MAX_HOLD_MS + 100,
  ],
  [
    "sem letras do fim do gesto, qualquer letra confiante solta (depois do mínimo)",
    () => {
      const h = createMovementHold();
      h.start({ letter: "J", confidence: 0.95, frames: [], times: [], endLetters: new Set() }, 0);
      return runUntilReleased(h, 6, () => same);
    },
    (r) => r !== null && r >= MIN_HOLD_MS,
  ],
];

let failures = 0;
for (const [name, run, check] of cases) {
  const r = run();
  const ok = check(r);
  if (!ok) failures++;
  console.log(`${ok ? "OK   " : "FALHA"} ${name} (saiu em ${r === null ? "nunca" : r + " ms"})`);
}
if (failures > 0) {
  console.error(`\n${failures} caso(s) falharam.`);
  process.exit(1);
}
console.log("\nTodos os casos passaram.");
