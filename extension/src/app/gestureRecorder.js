// Decide, quadro a quadro, se a mão está PARADA (então vale olhar a pose) ou
// fazendo um GESTO (então vale olhar o trajeto), e grava a sequência do gesto
// para o classificador de movimento.
//
// Sem isso, o classificador estático fica gritando letras erradas no meio de
// um gesto: durante um J, por exemplo, a mão passa por formas que parecem
// outras letras paradas.
//
// A velocidade é medida em "tamanhos de mão por segundo" — assim o limiar
// funciona igual com a mão perto ou longe da câmera.
//
// Estes números são um chute inicial e provavelmente precisam de ajuste ao
// vivo: se gestos não estiverem sendo detectados, baixe START_SPEED; se
// qualquer mexidinha virar gesto, suba.
const START_SPEED = 1.5; // acima disso: começou um gesto
const STOP_SPEED = 0.7; // abaixo disso: a mão está parando
const STILL_HOLD_MS = 250; // tempo parada para considerar o gesto encerrado
const MIN_GESTURE_MS = 300; // mais curto que isso é tremida, não gesto
const MAX_GESTURE_MS = 2500; // mais longo que isso: desiste e recomeça
const PRE_ROLL_MS = 200; // frames guardados ANTES do movimento disparar, para
// não perder o comecinho do gesto
const SMOOTHING = 0.7; // suavização da velocidade (0 = crua, perto de 1 = lenta)

const WRIST = 0;
const MIDDLE_MCP = 9;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

export function createGestureRecorder({ onGesture } = {}) {
  let state = "still"; // "still" | "moving"
  let preRoll = []; // { landmarks, t } recentes, enquanto parada
  let gesture = []; // { landmarks, t } do gesto em andamento
  let prev = null; // frame anterior, para calcular velocidade
  let speed = 0;
  let stillSince = null;
  let movementStartedAt = null; // quando o movimento disparou (depois do pre-roll)

  function reset() {
    state = "still";
    gesture = [];
    stillSince = null;
    movementStartedAt = null;
  }

  function finish(now) {
    // Corta os frames em que a mão já estava parada no fim: eles não fazem
    // parte do movimento, e se entrassem distorceriam tanto o trajeto quanto
    // a "forma da mão no fim". Também é por isso que a duração é medida até o
    // momento da parada, e não até agora — senão uma tremida de 100ms
    // contaria como um gesto de ~700ms (o tempo de confirmar que parou) e
    // passaria batido pelo filtro de duração mínima.
    //
    // A duração é contada a partir de quando o movimento realmente disparou,
    // não do início do buffer: os frames de pre-roll são contexto útil pro
    // classificador, mas somariam 200ms fixos a qualquer tremida.
    const stoppedAt = stillSince === null ? now : stillSince;
    const startedAt = movementStartedAt === null ? gesture[0].t : movementStartedAt;
    const active = gesture.filter((g) => g.t <= stoppedAt);
    const duration = stoppedAt - startedAt;
    reset();

    if (duration >= MIN_GESTURE_MS && active.length >= 2 && onGesture) {
      onGesture(
        active.map((g) => g.landmarks),
        active.map((g) => g.t)
      );
    }
  }

  // landmarks: os 21 pontos crus do frame atual, ou null se não há mão.
  // Devolve o estado atual ("still" | "moving").
  function update(landmarks, now) {
    if (!landmarks) {
      // mão saiu do quadro: descarta o que estava gravando
      preRoll = [];
      prev = null;
      speed = 0;
      reset();
      return state;
    }

    const handSize = Math.max(dist(landmarks[MIDDLE_MCP], landmarks[WRIST]), 1e-6);

    if (prev) {
      const dt = Math.max((now - prev.t) / 1000, 1e-3);
      const instant = dist(landmarks[WRIST], prev.landmarks[WRIST]) / handSize / dt;
      speed = SMOOTHING * speed + (1 - SMOOTHING) * instant;
    }
    prev = { landmarks, t: now };

    if (state === "still") {
      preRoll.push({ landmarks, t: now });
      while (preRoll.length > 1 && now - preRoll[0].t > PRE_ROLL_MS) preRoll.shift();

      if (speed > START_SPEED) {
        state = "moving";
        gesture = preRoll.slice(); // aproveita o pre-roll como início do gesto
        preRoll = [];
        stillSince = null;
        movementStartedAt = now;
      }
      return state;
    }

    // state === "moving"
    gesture.push({ landmarks, t: now });

    if (now - gesture[0].t > MAX_GESTURE_MS) {
      reset();
      return state;
    }

    if (speed < STOP_SPEED) {
      if (stillSince === null) stillSince = now;
      if (now - stillSince >= STILL_HOLD_MS) finish(now);
    } else {
      stillSince = null;
    }
    return state;
  }

  return { update, getState: () => state };
}
