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
import { createMovementHold } from "./movementHold.js";
import { createSmileDetector } from "./smileDetector.js";
import { playConfirmBeep } from "./sound.js";
import {
  addSample,
  addGesture,
  getCount,
  getGestureCount,
  downloadCsv,
} from "./sampleStore.js";

// O MediaPipe imprime avisos internos de diagnóstico via console.warn ("W1005
// 22:02:31 gl_context.cc:...", "INFO: Created TensorFlow Lite..."). Não são
// erros, mas o Chrome os lista na página de erros da extensão, o que esconde
// os erros de verdade. Filtramos só esse formato; qualquer outro aviso passa.
const MEDIAPIPE_LOG = /^(?:[IW]\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ [\w.]+:\d+\]|INFO: )/;
for (const level of ["warn", "info", "log"]) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    if (typeof args[0] === "string" && MEDIAPIPE_LOG.test(args[0])) return;
    original(...args);
  };
}

// A página de erros da extensão mostra DOMException como "[object
// DOMException]", sem dizer o que houve. Sempre registramos nome + mensagem.
function describeError(err) {
  return err && err.name ? `${err.name}: ${err.message}` : String(err);
}
window.addEventListener("unhandledrejection", (event) => {
  console.error(`Erro não tratado — ${describeError(event.reason)}`, event.reason);
});

const el = {
  status: document.getElementById("status"),
  stageState: document.getElementById("stage-state"),
  stageStateText: document.getElementById("stage-state-text"),
  letter: document.getElementById("letter"),
  letterCard: document.getElementById("letter-card"),
  letterSource: document.getElementById("letter-source"),
  meterFill: document.getElementById("meter-fill"),
  confidence: document.getElementById("confidence"),
  gestureLog: document.getElementById("gesture-log"),
  recordAs: document.getElementById("record-as"),
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
// Mais alto que o das letras paradas: aqui um falso positivo é pior, porque
// o resultado do gesto substitui a letra parada na tela.
const MOVEMENT_CONFIDENCE_THRESHOLD = 0.9;

// Quantos quadros do fim do gesto olhar para saber que letras o classificador
// de poses lê ali (ver movementHold.js) — ~0,25 s.
const END_POSE_FRAMES = 15;
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

// Coordenadas do MediaPipe vão de 0 a 1; espelhar na horizontal é x -> 1 - x.
function mirrorX(landmarks) {
  return landmarks.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
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
  // aguardado: se falhar (ex.: aba recarregada enquanto a câmera inicia), o
  // erro cai no try/catch de start() com mensagem legível, em vez de virar um
  // "[object DOMException]" solto no console
  await el.video.play();
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
  const poses = getCount();
  const gestures = getGestureCount();
  const parts = [];
  if (poses > 0) parts.push(`${poses} pose${poses === 1 ? "" : "s"}`);
  if (gestures > 0) parts.push(`${gestures} gesto${gestures === 1 ? "" : "s"}`);
  el.confirmCount.textContent =
    parts.length === 0 ? "Nenhuma amostra confirmada" : `${parts.join(" e ")} confirmado${poses + gestures === 1 ? "" : "s"}`;
  el.downloadBtn.disabled = poses + gestures === 0;
}

function wireControls() {
  el.clearWord.addEventListener("click", () => {
    word = "";
    renderWord();
  });
  el.copyWord.addEventListener("click", async () => {
    // a área de transferência recusa a cópia se a página não estiver em foco
    try {
      await navigator.clipboard.writeText(word);
      el.copyWord.textContent = "Copiado";
    } catch {
      el.copyWord.textContent = "Não copiou";
    }
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
    // O gesto reconhecido fica na tela enquanto a mão continuar na pose do fim
    // dele — ver movementHold.js para o porquê.
    const movementHold = createMovementHold();
    let confirmFlashUntil = 0;

    function handleConfirm() {
      if (!currentSource) return;

      playConfirmBeep();
      speakLetter(currentLetter);
      word += currentLetter;
      renderWord();
      confirmFlashUntil = performance.now() + CONFIRM_FLASH_MS;

      // Poses e gestos têm formatos de CSV diferentes (ver sampleStore.js): uma
      // pose é um quadro só; um gesto é a sequência inteira de quadros.
      if (currentSource === "static" && currentHandLandmarks) {
        addSample(currentHandLandmarks, currentLetter);
        updateConfirmCount();
      } else if (currentSource === "movement" && movementHold.current && !movementHold.current.saved) {
        const g = movementHold.current;
        addGesture(g.frames, g.times, g.letter);
        g.saved = true; // um sorriso repetido não duplica o gesto
        updateConfirmCount();
      }
    }

    const smileDetector = createSmileDetector({ onSmile: handleConfirm });

    // Última decisão sobre um gesto, mostrada na tela: um gesto descartado
    // antes sumia em silêncio, e era impossível saber por quê.
    function logGesture(text, ok) {
      // gravando ao vivo, a linha "gravado como X" tem prioridade sobre o que
      // o modelo achou do mesmo gesto, que viria logo em seguida
      if (recordedNow && !text.startsWith("gravado")) return;
      el.gestureLog.textContent = text;
      el.gestureLog.dataset.ok = String(ok);
    }

    // Modo de gravação ao vivo: todo gesto detectado é guardado com a letra
    // escolhida, na hora, sem sorrir e sem depender do que o modelo achou. É o
    // jeito de gravar no MESMO ambiente em que o reconhecimento acontece
    // (câmera e navegador a ~60 quadros/s) — os scripts de gravação do
    // training/ capturam a ~30 e não reproduzem isso.
    let recordedNow = false;

    const gestureRecorder = createGestureRecorder({
      onReject: (reason) => logGesture(`gesto descartado: ${reason}`, false),
      onGesture: (frames, times) => {
        recordedNow = false;
        const recordLabel = el.recordAs.value;
        if (recordLabel) {
          addGesture(frames, times, recordLabel);
          updateConfirmCount();
          recordedNow = true;
          logGesture(
            `gravado como ${recordLabel} (${getGestureCount()} gestos nesta sessão, ${frames.length} quadros)`,
            true
          );
        }
        if (!movementIsReady()) {
          logGesture("gesto detectado, mas o modelo de movimento não foi treinado", false);
          return;
        }
        const result = predictGesture(frames, times);
        const pct = result ? `${Math.round(result.confidence * 100)}%` : "";
        // NADA e pouca certeza não mexem no gesto que está na tela. Soltá-lo
        // num NADA seria errado: um gesto às vezes é cortado em dois pedaços,
        // e o segundo pode ser lido como NADA. Quando a mão realmente segue
        // em frente, quem solta é o "gesto em andamento" (movementHold.js).
        if (!result) return;
        if (result.letter === NONE_LABEL) {
          logGesture(`gesto lido como NADA (${pct}) — movimento à toa`, false);
          return;
        }
        if (result.confidence < MOVEMENT_CONFIDENCE_THRESHOLD) {
          logGesture(
            `gesto lido como ${result.letter} com só ${pct} de certeza (precisa de ${Math.round(MOVEMENT_CONFIDENCE_THRESHOLD * 100)}%) — ignorado`,
            false
          );
          return;
        }
        logGesture(`gesto ${result.letter} reconhecido (${pct}, ${frames.length} quadros)`, true);

        // letras que o classificador de poses lê no FIM do gesto: é o que ele
        // vai mostrar assim que o gesto sair da tela, e não deve substituí-lo
        const endLetters = new Set();
        if (staticIsReady()) {
          for (const frame of frames.slice(-END_POSE_FRAMES)) {
            const p = predict(frame);
            if (p && p.confidence >= 0.5) endLetters.add(p.letter);
          }
        }
        movementHold.start(
          {
            letter: result.letter,
            confidence: result.confidence,
            frames, // guardados para virar amostra se você confirmar com o sorriso
            times,
            endLetters,
          },
          performance.now()
        );
        // já foi guardado pelo modo de gravação: um sorriso depois não duplica
        if (recordedNow && movementHold.current) movementHold.current.saved = true;
      },
    });

    let lastVideoTime = -1;
    function renderLoop() {
      if (el.video.currentTime !== lastVideoTime) {
        lastVideoTime = el.video.currentTime;
        const nowMs = performance.now();

        const handResult = handLandmarker.detectForVideo(el.video, nowMs);
        drawLandmarks(handResult.landmarks);
        // Os modelos recebem a mão ESPELHADA, porque é assim que os scripts de
        // gravação (training/collect_*.py) a enxergam: eles espelham a imagem
        // da webcam antes de detectar a mão, como num espelho. Aqui o espelho é
        // só visual (CSS), então a detecção chega com o x invertido em relação
        // ao treino. Sem esta linha, os modelos viam a "outra mão" — o H
        // reconhecido nas gravações não era reconhecido ao vivo. O desenho na
        // tela continua usando os pontos originais (o canvas já é espelhado).
        const hand = handResult.landmarks.length > 0 ? mirrorX(handResult.landmarks[0]) : null;

        // O gravador observa o movimento em segundo plano. Uma tremida NÃO
        // apaga a letra parada: ninguém segura uma letra 100% imóvel. Mas um
        // gesto de verdade em andamento (a mão já foi longe do ponto de
        // partida) sim: no meio do Z o classificador de poses chutava letras
        // (L, G, N, M...) em ~75% dos quadros, trocando de letra quase 4 vezes
        // por gesto — e um sorriso nessa hora confirmaria uma delas. O
        // resultado do gesto assume a tela quando o classificador de
        // movimento tem certeza.
        const motionState = gestureRecorder.update(hand, nowMs);

        let letter = NO_LETTER;
        let source = null;
        let confidence = 0;

        const inProgress = gestureRecorder.isGestureInProgress();
        const prediction = hand && staticIsReady() ? predict(hand) : null;
        const held = movementHold.update(
          {
            now: nowMs,
            hasHand: Boolean(hand),
            gestureInProgress: inProgress,
            staticLetter: prediction ? prediction.letter : null,
            staticConfidence: prediction ? prediction.confidence : 0,
          },
          CONFIDENCE_THRESHOLD
        );

        if (held) {
          letter = held.letter;
          confidence = held.confidence;
          source = "movement";
        } else if (prediction && !inProgress) {
          confidence = prediction.confidence;
          if (prediction.confidence >= CONFIDENCE_THRESHOLD) {
            letter = prediction.letter;
            source = "static";
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
        } else if (gestureRecorder.isGestureInProgress()) {
          setStageState("gesture", "Gesto em andamento");
        } else if (motionState === "moving") {
          setStageState("gesture", "Mão em movimento");
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
    console.error(describeError(err), err);
    setStatus(`Erro: ${err.message}`);
    setStageState("idle", "Erro");
  }
}

start();
