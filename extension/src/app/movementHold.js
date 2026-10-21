// Mantém na tela a letra de um gesto reconhecido (H, J, K, X, Z) pelo tempo
// certo.
//
// Um tempo fixo não serve: a mão termina o gesto numa pose qualquer, e o
// classificador de poses a lê como OUTRA letra. No Z, por exemplo, a mão
// termina numa pose que ele lê como N (e M) — então, passado o tempo fixo, a
// tela trocava o Z por N, mesmo com o Z reconhecido certo.
//
// Por isso a regra é: o gesto fica na tela pelo menos MIN_HOLD_MS (tempo de
// sorrir pra confirmar) e, depois disso, ENQUANTO a mão continuar lendo as
// letras que ela já lia no fim do gesto. Solta quando:
//   - aparece uma letra confiante DIFERENTE das do fim do gesto (você mudou de
//     pose — a distância entre poses não separa "mesma pose" de "pose
//     parecida", então a regra olha a letra lida, não a distância);
//   - a mão sai do quadro;
//   - passa de MAX_HOLD_MS (rede de segurança: nunca prende a tela pra sempre).
export const MIN_HOLD_MS = 1800;
export const MAX_HOLD_MS = 8000;
const RELEASE_FRAMES = 6; // quadros seguidos com uma letra nova, p/ ignorar um piscar

export function createMovementHold() {
  let hold = null; // { letter, confidence, frames, times, saved, endLetters, since, minUntil, maxUntil }
  let newLetterRun = 0;

  return {
    // endLetters: Set das letras que o classificador de poses leu nos últimos
    // quadros do gesto
    start({ letter, confidence, frames, times, endLetters }, now) {
      hold = {
        letter,
        confidence,
        frames,
        times,
        saved: false,
        endLetters: new Set(endLetters),
        minUntil: now + MIN_HOLD_MS,
        maxUntil: now + MAX_HOLD_MS,
      };
      newLetterRun = 0;
    },

    release() {
      hold = null;
      newLetterRun = 0;
    },

    // Chamar a cada quadro. staticLetter/staticConfidence: o que o classificador
    // de poses lê agora (ou null). Devolve o gesto ativo, ou null.
    update({ now, hasHand, gestureInProgress, staticLetter, staticConfidence }, confidentThreshold) {
      if (!hold) return null;

      if (!hasHand || now >= hold.maxUntil) {
        this.release();
        return null;
      }

      // Mão se mexendo de novo: o gesto continua na tela e as letras paradas
      // não são avaliadas (no meio do movimento elas são lixo). Soltar aqui
      // seria errado: um gesto às vezes é cortado em dois pedaços, e o segundo
      // pedaço não pode derrubar o primeiro já reconhecido. Se a mão seguiu
      // em frente de verdade, a mudança de pose solta logo depois.
      if (gestureInProgress) {
        newLetterRun = 0;
        return hold;
      }

      if (now >= hold.minUntil) {
        const isNewLetter =
          staticLetter && staticConfidence >= confidentThreshold && !hold.endLetters.has(staticLetter);
        newLetterRun = isNewLetter ? newLetterRun + 1 : 0;
        if (newLetterRun >= RELEASE_FRAMES) {
          this.release();
          return null;
        }
      }
      return hold;
    },

    get current() {
      return hold;
    },
  };
}
