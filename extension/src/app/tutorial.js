// O pop-up do pinguim: ao clicar numa letra do alfabeto, mostra a mão em
// esqueleto fazendo aquela letra, repetindo em loop.
//
// A mão mostrada é uma gravação do autor do projeto, escolhida por
// training/export_tutorial.py (a amostra mais típica de cada letra). Não é
// material oficial de Libras, e o pop-up diz isso.
//
// Os pontos estão na orientação de ESPELHO, a mesma do vídeo na tela: quem
// olha o tutorial e depois a câmera vê a mão do mesmo lado.
import { loadModelFile } from "../classifier/mlp.js";
import { HAND_CONNECTIONS } from "./hand.js";

const REFS_PATH = "lib/models/tutorial_refs.json";
// x e y do MediaPipe são frações da largura e da altura de um vídeo 4:3. Sem
// esticar o x, a mão desenhada sairia achatada.
const ASPECT = 4 / 3;
const PADDING = 0.14; // margem em volta da mão, em fração do quadro
const START_PAUSE_MS = 600; // parado na pose inicial antes de mexer
const END_PAUSE_MS = 1000; // parado na pose final antes de recomeçar
const SLOW_FACTOR = 2.5; // "ver devagar": quantas vezes mais lento
const FINGERTIPS = [4, 8, 12, 16, 20];

const TRACK = "#4fd1e0";
const BEAK = "#f2903f";

let refs = { poses: {}, gestures: {} };

// Sem o arquivo (ainda não gerado), o pop-up abre e explica como gerar.
export async function loadTutorialRefs() {
  const data = await loadModelFile(REFS_PATH);
  if (data) refs = data;
}

// Ponta de dedo que mais anda no gesto: é o rastro desenhado, porque é o
// que "escreve" a letra no ar (o indicador no Z, o mínimo no J...).
function busiestFingertip(frames) {
  let best = FINGERTIPS[0];
  let bestLength = -1;
  for (const tip of FINGERTIPS) {
    let length = 0;
    for (let i = 1; i < frames.length; i++) {
      length += Math.hypot(frames[i][tip][0] - frames[i - 1][tip][0], frames[i][tip][1] - frames[i - 1][tip][1]);
    }
    if (length > bestLength) {
      best = tip;
      bestLength = length;
    }
  }
  return best;
}

// Escala e posição que fazem TODOS os quadros caberem no canvas sem mudar de
// tamanho no meio da animação.
function fitTransform(frames, width, height) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const frame of frames) {
    for (const [x, y] of frame) {
      minX = Math.min(minX, x * ASPECT);
      maxX = Math.max(maxX, x * ASPECT);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  const usableW = width * (1 - 2 * PADDING);
  const usableH = height * (1 - 2 * PADDING);
  const scale = Math.min(usableW / Math.max(maxX - minX, 1e-6), usableH / Math.max(maxY - minY, 1e-6));
  const offsetX = (width - (maxX - minX) * scale) / 2 - minX * scale;
  const offsetY = (height - (maxY - minY) * scale) / 2 - minY * scale;
  return ([x, y]) => [x * ASPECT * scale + offsetX, y * scale + offsetY];
}

function lerpFrame(a, b, t) {
  return a.map(([x, y], i) => [x + (b[i][0] - x) * t, y + (b[i][1] - y) * t]);
}

export function createTutorial(el) {
  const ctx = el.canvas.getContext("2d");
  let animation = null;

  function sizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    el.canvas.width = Math.round(el.canvas.clientWidth * ratio);
    el.canvas.height = Math.round(el.canvas.clientHeight * ratio);
  }

  function drawHand(points, ratio) {
    ctx.lineCap = "round";
    ctx.strokeStyle = TRACK;
    ctx.lineWidth = 3 * ratio;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(...points[a]);
      ctx.lineTo(...points[b]);
      ctx.stroke();
    }
    ctx.fillStyle = BEAK;
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x, y, 4.5 * ratio, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function drawTrail(path, upTo, ratio) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // o caminho inteiro, apagado, e por cima o trecho já percorrido
    for (const [end, alpha, dash] of [[path.length - 1, 0.25, [6 * ratio, 7 * ratio]], [upTo, 0.9, []]]) {
      if (end < 1) continue;
      ctx.globalAlpha = alpha;
      ctx.setLineDash(dash);
      ctx.strokeStyle = BEAK;
      ctx.lineWidth = 3 * ratio;
      ctx.beginPath();
      ctx.moveTo(...path[0]);
      for (let i = 1; i <= end; i++) ctx.lineTo(...path[i]);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }

  function stop() {
    if (animation) cancelAnimationFrame(animation);
    animation = null;
  }

  function playPose(pose) {
    sizeCanvas();
    const ratio = window.devicePixelRatio || 1;
    const toCanvas = fitTransform([pose], el.canvas.width, el.canvas.height);
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    drawHand(pose.map(toCanvas), ratio);
  }

  function playGesture(gesture) {
    sizeCanvas();
    const ratio = window.devicePixelRatio || 1;
    const { frames, frame_ms: frameMs } = gesture;
    const toCanvas = fitTransform(frames, el.canvas.width, el.canvas.height);
    const canvasFrames = frames.map((f) => f.map(toCanvas));
    const tip = busiestFingertip(frames);
    const trail = canvasFrames.map((f) => f[tip]);
    const startedAt = performance.now();

    function tick(now) {
      const speed = el.slow.checked ? SLOW_FACTOR : 1;
      const moveMs = (frames.length - 1) * frameMs * speed;
      const cycle = START_PAUSE_MS + moveMs + END_PAUSE_MS;
      const t = (now - startedAt) % cycle;
      const progress = Math.min(Math.max((t - START_PAUSE_MS) / moveMs, 0), 1);
      const position = progress * (frames.length - 1);
      const i = Math.floor(position);
      const current = i >= frames.length - 1
        ? canvasFrames[frames.length - 1]
        : lerpFrame(canvasFrames[i], canvasFrames[i + 1], position - i);

      ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
      drawTrail(trail, i, ratio);
      drawHand(current, ratio);
      animation = requestAnimationFrame(tick);
    }
    // o primeiro quadro sai na hora, sem esperar o navegador pedir o próximo
    tick(startedAt);
  }

  function open(letter) {
    stop();
    el.title.textContent = `Como fazer o ${letter}`;
    const gesture = refs.gestures[letter];
    const pose = refs.poses[letter];
    el.slowWrap.hidden = !gesture;
    el.note.hidden = !(gesture || pose);
    el.dialog.showModal();
    // o canvas só tem tamanho depois que o pop-up aparece na tela
    if (gesture) {
      el.say.textContent = "Comece nesta forma, faça o movimento e pare a mão no final. Depois é só sorrir!";
      playGesture(gesture);
    } else if (pose) {
      el.say.textContent = "Faça esta forma com a mão e segure parada. Depois é só sorrir!";
      playPose(pose);
    } else {
      el.say.textContent = "Ainda não tenho uma gravação desta letra para te mostrar.";
      ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    }
  }

  el.dialog.addEventListener("close", stop);
  el.close.addEventListener("click", () => el.dialog.close());
  // clicar fora do cartão (no fundo escurecido) também fecha
  el.dialog.addEventListener("click", (event) => {
    if (event.target === el.dialog) el.dialog.close();
  });

  return { open };
}
