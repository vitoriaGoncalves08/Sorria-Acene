// Confere que movementFeatures.js (JavaScript) calcula exatamente as mesmas
// características que training/movement_features.py (Python).
//
// Se os dois saírem de sincronia, o modelo treinado em Python recebe entradas
// diferentes aqui na extensão e passa a errar SEM DAR ERRO NENHUM — só piora
// a acurácia em silêncio. Por isso este teste existe.
//
// Uso:
//   cd training && .venv\Scripts\python.exe check_parity.py   (gera o fixture)
//   cd extension && npm test
import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const featuresPath = path.join(here, "..", "src", "classifier", "movementFeatures.js");
const fixturePath = path.join(here, "..", "..", "training", "data", "parity_fixture.json");

const TOLERANCE = 1e-9;

if (!fs.existsSync(fixturePath)) {
  console.error(
    `Fixture não encontrado em ${fixturePath}\n` +
      "Rode primeiro:  cd training && .venv\\Scripts\\python.exe check_parity.py"
  );
  process.exit(1);
}

const { extractFeatures } = await import(pathToFileURL(featuresPath).href);
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

const got = extractFeatures(
  fixture.frames.map((frame) => frame.map((p) => ({ x: p[0], y: p[1], z: p[2] }))),
  fixture.timesMs
);
const want = fixture.expected;

if (!got) {
  console.error("FALHOU: extractFeatures devolveu null");
  process.exit(1);
}
if (got.length !== want.length) {
  console.error(`FALHOU: JavaScript devolveu ${got.length} números, Python ${want.length}`);
  process.exit(1);
}

let maxDiff = 0;
const diffs = [];
for (let i = 0; i < want.length; i++) {
  const d = Math.abs(got[i] - want[i]);
  if (d > maxDiff) maxDiff = d;
  if (d > TOLERANCE) diffs.push({ i, want: want[i], got: got[i], d });
}

if (diffs.length > 0) {
  console.error(`FALHOU: ${diffs.length} de ${want.length} características divergem:`);
  for (const r of diffs.slice(0, 10)) {
    console.error(`  [${r.i}] python=${r.want} javascript=${r.got} diff=${r.d.toExponential(2)}`);
  }
  process.exit(1);
}

console.log(
  `OK — as ${want.length} características batem entre Python e JavaScript ` +
    `(maior diferença: ${maxDiff.toExponential(2)}, só ruído de ponto flutuante)`
);
