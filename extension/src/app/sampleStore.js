// Acumula, em memória, as amostras confirmadas por sorriso durante a
// sessão, e permite baixá-las como CSV — no MESMO formato de
// training/data/landmarks.csv (p0..p62,label), pra você poder incorporar
// esses dados manualmente no treino depois. Nada é enviado a servidor
// nenhum (a extensão não tem backend nesta fase) nem gravado em disco sem
// você clicar em "baixar".
import { normalizeLandmarks } from "../classifier/classifier.js";

const samples = [];

export function addSample(handLandmarks, letter) {
  samples.push({ features: normalizeLandmarks(handLandmarks), letter });
}

export function getCount() {
  return samples.length;
}

export function downloadCsv() {
  if (samples.length === 0) return;

  const header = Array.from({ length: 63 }, (_, i) => `p${i}`).concat("label").join(",");
  const rows = samples.map((s) => s.features.concat(s.letter).join(","));
  const csv = [header, ...rows].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `amostras_confirmadas_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
