import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "../../lib/mediapipe/vision_bundle.mjs";
import {
  loadModel as loadStaticModel,
  isReady as staticIsReady,
  getLabels as staticLabels,
  predict,
} from "../classifier/classifier.js";
import {
  loadModel as loadMovementModel,
  isReady as movementIsReady,
  getLabels as movementLabels,
  predictGesture,
  NONE_LABEL,
} from "../classifier/movementClassifier.js";
import { createGestureRecorder } from "./gestureRecorder.js";
import { createSmileDetector } from "./smileDetector.js";
import { playConfirmBeep } from "./sound.js";
import { addSample, getCount, downloadCsv } from "./sampleStore.js";

const el = {
  status: document.getElementById("status"),
  stageState: document.getElementById("stage-state"),
  stageStateText: document.getElementById("stage-state-text"),
  letter: document.getElementById("letter"),
  letterCard: document.getElementById("letter-card"),
  letterSource: document.getElementById("letter-source"),
  meterFill: document.getElementById("meter-fill"),
  confidence: document.getElementById("confidence"),
  word: document.getElementById("word"),
  copyWord: document.getElementById("copy-word"),
  clearWord: document.getElementById("clear-word"),
  rail: document.getElementById("rail"),
  speechToggle: document.getElementById("speech-toggle"),
  confirmCount: document.getElementById("confirm-count"),
  downloadBtn: document.getElementById("download-samples"),
  video: document.getElementById("webcam"),
  canvas: document.getElementById("overlay"),
};
const ctx = el.canvas.getContext("2d");

// Abaixo disso, o classificador está "chutando" — melhor não mostrar letra
// nenhuma do que mostrar uma errada com confiança baixa.
const CONFIDENCE_THRESHOLD = 0.8;
const MOVEMENT_CONFIDENCE_THRESHOLD = 0.7;

// Depois de reconhecer uma letra com movimento, ela fica na tela por esse
// tempo — senão o classificador estático assumiria de volta no quadro
// seguinte (a mão continua parada em alguma pose no fim do gesto) e você não
// teria tempo de sorrir pra confirmar.
const MOVEMENT_HOLD_MS = 1800;
const CONFIRM_FLASH_MS = 450;

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const NO_LETTER = "—";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // polegar
  [0, 5], [5, 6], [6, 7], [7, 8], // indicador
  [5, 9], [9, 10], [10, 11], [11, 12], // médio
  [9, 13], [13, 14], [14, 15], [15, 16], // anelar
  [13, 17], [17, 18], [18, 19], [19, 20], // mínimo
  [0, 17],
];

function setStatus(text) {
  el.status.textContent = text;
}

function setStageState(state, text) {
  el.stageState.dataset.state = state;
  el.stageStateText.textContent = text;
}

// Só é chamada depois de um sorriso de confirmação — por isso fala sempre que
// chamada, sem dedupe por letra: o sorriso já é o gatilho.
function speakLetter(letter) {
  if (!el.speechToggle.checked) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(letter);
  utterance.lang = "pt-BR";
  speechSynthesis.speak(utterance);
}

function drawLandmarks(landmarksList) {
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
  for (const landmarks of landmarksList) {
    ctx.strokeStyle = "#4fd1e0";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = landmarks[a];
      const pb = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(pa.x * el.canvas.width, pa.y * el.canvas.height);
      ctx.lineTo(pb.x * el.canvas.width, pb.y * el.canvas.height);
      ctx.stroke();
    }
    ctx.fillStyle = "#f2903f";
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * el.canvas.width, p.y * el.canvas.height, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  el.video.srcObject = stream;
  await new Promise((resolve) => {
    el.video.onloadedmetadata = () => resolve();
  });
  el.video.play();
  el.canvas.width = el.video.videoWidth;
  el.canvas.height = el.video.videoHeight;
}

async function setupHandLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    chrome.runtime.getURL("lib/mediapipe/wasm")
  );
  return HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: chrome.runtime.getURL("lib/models/hand_landmarker.task"),
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

// Igual ao HandLandmarker, mas resiliente: se o modelo de rosto ainda não foi
// baixado (ver README.md), a extensão continua reconhecendo letras — só a
// confirmação por sorriso fica indisponível.
async function setupFaceLandmarker() {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      chrome.runtime.getURL("lib/mediapipe/wasm")
    );
    return await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: chrome.runtime.getURL("lib/models/face_landmarker.task"),
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
    });
  } catch (err) {
    console.warn("Confirmação por sorriso indisponível:", err.message);
    return null;
  }
}

// O alfabeto na tela é montado a partir do que os modelos REALMENTE conhecem,
// então ele nunca fica mentindo sobre o que o app sabe fazer.
const slots = new Map();
function buildRail() {
  const poses = new Set(staticLabels());
  const moves = new Set(movementLabels());
  el.rail.replaceChildren();

  for (const letter of ALPHABET) {
    const kind = poses.has(letter) ? "pose" : moves.has(letter) ? "move" : "off";
    const slot = document.createElement("div");
    slot.className = "slot";
    slot.dataset.kind = kind;
    slot.textContent = letter;
    slot.title =
      kind === "pose"
        ? `${letter} — pose parada`
        : kind === "move"
          ? `${letter} — precisa de movimento`
          : `${letter} — ainda sem treino`;
    el.rail.append(slot);
    slots.set(letter, slot);
  }
}

function highlightRail(letter) {
  for (const [key, slot] of slots) {
    slot.dataset.active = String(key === letter);
  }
}

// A palavra sendo soletrada: cada letra confirmada por sorriso entra aqui.
let word = "";

function renderWord() {
  el.word.textContent = word;
  el.copyWord.disabled = word.length === 0;
  el.clearWord.disabled = word.length === 0;
  for (const [key, slot] of slots) {
    slot.dataset.used = String(word.includes(key));
  }
}

function updateConfirmCount() {
  const count = getCount();
  el.confirmCount.textContent =
    count === 0
      ? "Nenhuma amostra confirmada"
      : `${count} amostra${count === 1 ? "" : "s"} confirmada${count === 1 ? "" : "s"}`;
  el.downloadBtn.disabled = count === 0;
}

function wireControls() {
  el.clearWord.addEventListener("click", () => {
    word = "";
    renderWord();
  });
  el.copyWord.addEventListener("click", async () => {
    await navigator.clipboard.writeText(word);
    el.copyWord.textContent = "Copiado";
    setTimeout(() => (el.copyWord.textContent = "Copiar"), 1200);
  });
  el.downloadBtn.addEventListener("click", downloadCsv);
}

async function start() {
  buildRail();
  wireControls();
  renderWord();
  updateConfirmCount();

  try {
    setStatus("Pedindo acesso à webcam…");
    await setupCamera();

    setStatus("Carregando detecção de mãos…");
    const handLandmarker = await setupHandLandmarker();

    setStatus("Carregando detecção de rosto…");
    const faceLandmarker = await setupFaceLandmarker();

    setStatus("Carregando classificadores…");
    await loadStaticModel();
    await loadMovementModel();
    buildRail(); // agora com as letras que os modelos de fato conhecem
    renderWord();

    if (!staticIsReady()) {
      setStatus("Classificador de letras ainda não treinado — veja training/README.md.");
    } else if (!faceLandmarker) {
      setStatus("Sem confirmação por sorriso: falta lib/models/face_landmarker.task.");
    } else if (!movementIsReady()) {
      setStatus("H, J, K, X e Z ainda não foram treinadas — veja training/README.md.");
    } else {
      setStatus("Faça uma letra e sorria para confirmar.");
    }

    // Estado "tentativo" do quadro atual — a confirmação por sorriso usa
    // esses valores no instante em que o sorriso é reconhecido.
    let currentLetter = NO_LETTER;
    let currentHandLandmarks = null;
    let currentSource = null; // "static" | "movement" | null
    let movementHold = null;
    let confirmFlashUntil = 0;

    function handleConfirm() {
      if (!currentSource) return;

      playConfirmBeep();
      speakLetter(currentLetter);
      word += currentLetter;
      renderWord();
      confirmFlashUntil = performance.now() + CONFIRM_FLASH_MS;

      // Só letras PARADAS viram amostra: o CSV tem o formato de uma pose única
      // (63 números de um quadro só), que não representa um gesto.
      if (currentSource === "static" && currentHandLandmarks) {
        addSample(currentHandLandmarks, currentLetter);
        updateConfirmCount();
      }
    }

    const smileDetector = createSmileDetector({ onSmile: handleConfirm });

    const gestureRecorder = createGestureRecorder({
      onGesture: (frames, times) => {
        if (!movementIsReady()) return;
        const result = predictGesture(frames, times);
        if (!result || result.letter === NONE_LABEL) return;
        if (result.confidence < MOVEMENT_CONFIDENCE_THRESHOLD) return;
        movementHold = {
          letter: result.letter,
          confidence: result.confidence,
          until: performance.now() + MOVEMENT_HOLD_MS,
        };
      },
    });

    let lastVideoTime = -1;
    function renderLoop() {
      if (el.video.currentTime !== lastVideoTime) {
        lastVideoTime = el.video.currentTime;
        const nowMs = performance.now();

        const handResult = handLandmarker.detectForVideo(el.video, nowMs);
        drawLandmarks(handResult.landmarks);
        const hand = handResult.landmarks.length > 0 ? handResult.landmarks[0] : null;

        // Mão parada é pose; mão em movimento é gesto. Sem essa arbitragem, o
        // classificador estático gritaria letras erradas no meio de um gesto.
        const motionState = gestureRecorder.update(hand, nowMs);

        let letter = NO_LETTER;
        let source = null;
        let confidence = 0;

        if (movementHold && nowMs < movementHold.until) {
          letter = movementHold.letter;
          confidence = movementHold.confidence;
          source = "movement";
        } else {
          movementHold = null;
          if (motionState === "moving") {
            letter = NO_LETTER;
          } else if (hand && staticIsReady()) {
            const prediction = predict(hand);
            if (prediction) {
              confidence = prediction.confidence;
              if (prediction.confidence >= CONFIDENCE_THRESHOLD) {
                letter = prediction.letter;
                source = "static";
              }
            }
          }
        }

        currentLetter = letter;
        currentSource = source;
        currentHandLandmarks = hand;

        // --- pintura ---
        el.letter.textContent = letter;
        el.letter.dataset.empty = String(source === null);
        el.letterCard.dataset.confirmed = String(nowMs < confirmFlashUntil);
        el.letterSource.textContent =
          source === "movement" ? "movimento" : source === "static" ? "pose" : "";

        el.meterFill.style.width = `${Math.round(confidence * 100)}%`;
        el.confidence.textContent = source
          ? `${Math.round(confidence * 100)}% de confiança`
          : hand
            ? "analisando…"
            : "sem leitura";

        highlightRail(source ? letter : null);

        if (!hand) {
          setStageState("idle", "Mostre a mão para a câmera");
        } else if (motionState === "moving") {
          setStageState("gesture", "Gravando gesto…");
        } else {
          setStageState("tracking", "Mão detectada");
        }

        if (faceLandmarker) {
          const faceResult = faceLandmarker.detectForVideo(el.video, nowMs);
          smileDetector.update(faceResult.faceBlendshapes[0], nowMs);
        }
      }
      requestAnimationFrame(renderLoop);
    }
    renderLoop();
  } catch (err) {
    console.error(err);
    setStatus(`Erro: ${err.message}`);
    setStageState("idle", "Erro");
  }
}

start();
