// Acumula, em memória, as amostras confirmadas por sorriso durante a sessão e
// permite baixá-las como CSV, pra você incorporar no treino depois. Nada é
// enviado a servidor nenhum (a extensão não tem backend nesta fase) nem
// gravado em disco sem você clicar em "baixar".
//
// São dois formatos, porque há dois tipos de letra:
//  - poses (letras paradas):  p0..p62,label  — igual a training/data/landmarks.csv
//  - gestos (H, J, K, X, Z):  sequence_id,frame,t_ms,x0,y0,z0..z20,label — igual
//    a training/data/movement_sequences.csv, com os landmarks CRUS de cada
//    quadro (as características são calculadas só no treino).
import { normalizeLandmarks } from "../classifier/classifier.js";

const samples = [];
const gestures = [];

export function addSample(handLandmarks, letter) {
  samples.push({ features: normalizeLandmarks(handLandmarks), letter });
}

// frames: array de quadros, cada um com 21 pontos {x,y,z}, já na orientação
// espelhada que os modelos usam (a mesma de collect_movement_data.py).
export function addGesture(frames, timesMs, letter) {
  gestures.push({
    frames: frames.map((f) => f.map((p) => [p.x, p.y, p.z])),
    times: [...timesMs],
    letter,
  });
}

export function getCount() {
  return samples.length;
}

export function getGestureCount() {
  return gestures.length;
}

function download(filename, text) {
  const blob = new Blob([text], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function posesCsv() {
  const header = Array.from({ length: 63 }, (_, i) => `p${i}`).concat("label").join(",");
  const rows = samples.map((s) => s.features.concat(s.letter).join(","));
  return [header, ...rows].join("\n");
}

function gesturesCsv() {
  const header = ["sequence_id", "frame", "t_ms"];
  for (let i = 0; i < 21; i++) header.push(`x${i}`, `y${i}`, `z${i}`);
  header.push("label");

  const base = Date.now();
  const rows = [];
  gestures.forEach((g, n) => {
    g.frames.forEach((frame, i) => {
      rows.push([base + n, i, g.times[i], ...frame.flat(), g.letter].join(","));
    });
  });
  return [header.join(","), ...rows].join("\n");
}

// Baixa um arquivo para cada tipo que tiver amostras.
export function downloadCsv() {
  const stamp = Date.now();
  if (samples.length > 0) download(`amostras_confirmadas_${stamp}.csv`, posesCsv());
  if (gestures.length > 0) {
    // pequena folga: alguns navegadores bloqueiam dois downloads no mesmo instante
    setTimeout(() => download(`gestos_confirmados_${stamp}.csv`, gesturesCsv()), 300);
  }
}
