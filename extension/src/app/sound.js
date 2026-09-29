// Bipe curto de confirmação, sintetizado na hora (sem arquivo de áudio).
let audioCtx = null;

function getContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function playTone(ctx, freq, startTime, duration) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  // rampa de volume em vez de ligar/desligar seco, pra não estalar
  gain.gain.setValueAtTime(0, startTime);
  gain.gain.linearRampToValueAtTime(0.2, startTime + 0.01);
  gain.gain.linearRampToValueAtTime(0, startTime + duration);
  osc.connect(gain).connect(ctx.destination);
  osc.start(startTime);
  osc.stop(startTime + duration);
}

export function playConfirmBeep() {
  const ctx = getContext();
  const now = ctx.currentTime;
  playTone(ctx, 880, now, 0.09); // duas notas subindo, tipo "ding-ding" de acerto
  playTone(ctx, 1320, now + 0.09, 0.12);
}
