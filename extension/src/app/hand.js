// Os ossos da mão: pares de pontos do MediaPipe ligados por uma linha no
// desenho. Usado tanto no vídeo ao vivo quanto no tutorial.
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // polegar
  [0, 5], [5, 6], [6, 7], [7, 8], // indicador
  [5, 9], [9, 10], [10, 11], [11, 12], // médio
  [9, 13], [13, 14], [14, 15], [15, 16], // anelar
  [13, 17], [17, 18], [18, 19], [19, 20], // mínimo
  [0, 17],
];
