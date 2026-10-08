// As características do gesto não podem depender da taxa de quadros.
//
// Aconteceu de verdade: gestos gravados por script (~30 fps) tinham ~2
// inversões de direção, e os mesmos gestos feitos na extensão (~60 fps)
// chegavam a 7, porque cada tremidinha do landmark virava uma inversão. O
// modelo treinava com um e recebia o outro.
//
// O teste simula o mesmo sinal da câmera visto a 60 fps e a 30 fps (metade dos
// quadros do MESMO sinal) e exige que as características coincidam.
//
// Uso: cd extension && npm test
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { extractFeatures } = await import(
  pathToFileURL(path.join(here, "..", "src", "classifier", "movementFeatures.js")).href
);

// gerador pseudoaleatório determinístico (o teste não pode variar entre rodadas)
function prng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

// 1,5 s a 60 fps: uma mão quase parada com leve deriva para a direita e um
// ruído de landmark por quadro MAIOR que a zona morta de inversões — o caso
// em que cada quadro extra vira uma inversão falsa. Começa e termina em
// instantes pares, para o recorte de 30 fps cobrir exatamente o mesmo
// intervalo.
function camera60() {
  const rand = prng(7);
  const frames = [];
  const times = [];
  for (let i = 0; i <= 90; i++) {
    const t = i / 60;
    const x = 0.3 + 0.03 * (t / 1.5) + 0.006 * rand();
    const y = 0.4;
    const frame = [];
    for (let p = 0; p < 21; p++) {
      frame.push({ x: x + 0.01 * p, y: y + 0.008 * ((p * 7) % 11), z: 0.001 * p });
    }
    frame[0] = { x, y, z: 0 };
    frame[9] = { x, y: y + 0.1, z: 0 }; // régua da mão
    frames.push(frame);
    times.push(t * 1000);
  }
  return { frames, times };
}

const REVERSALS = 58; // posições no vetor de 60 características
const STRAIGHTNESS = 57;

const at60 = camera60();
const at30 = {
  frames: at60.frames.filter((_, i) => i % 2 === 0),
  times: at60.times.filter((_, i) => i % 2 === 0),
};
const f60 = extractFeatures(at60.frames, at60.times);
const f30 = extractFeatures(at30.frames, at30.times);

let failures = 0;
function check(name, v30, v60, tolerance) {
  const ok = Math.abs(v30 - v60) <= tolerance;
  if (!ok) failures++;
  console.log(`${ok ? "OK   " : "FALHA"} ${name}: 30 fps = ${v30.toFixed(2)} | 60 fps = ${v60.toFixed(2)}`);
}

check("inversões de direção iguais em 30 e 60 fps", f30[REVERSALS], f60[REVERSALS], 0);
check("retilineidade igual em 30 e 60 fps", f30[STRAIGHTNESS], f60[STRAIGHTNESS], 0.05);

if (failures > 0) {
  console.error(`\n${failures} caso(s) falharam.`);
  process.exit(1);
}
console.log("\nCaracterísticas independentes da taxa de quadros.");
