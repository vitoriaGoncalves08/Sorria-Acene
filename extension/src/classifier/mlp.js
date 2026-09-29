// Aplica uma rede neural pequena exportada pelo treino em Python
// (training/export_model.py) — o formato é sempre:
//   { labels: string[], layers: [{ weights, biases, activation }] }
//
// Usado tanto pelo classificador estático (classifier.js) quanto pelo de
// movimento (movementClassifier.js): os dois têm entradas e classes
// diferentes, mas a conta é exatamente a mesma.

function relu(v) {
  return v.map((x) => Math.max(0, x));
}

function softmax(v) {
  const max = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function denseForward(input, weights, biases) {
  // weights: n_entradas x n_saidas | biases: n_saidas
  const numOutputs = biases.length;
  const output = new Array(numOutputs).fill(0);
  for (let j = 0; j < numOutputs; j++) {
    let sum = biases[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * weights[i][j];
    }
    output[j] = sum;
  }
  return output;
}

// Devolve { label, confidence } — a classe de maior probabilidade e o quanto
// o modelo confia nela.
export function forwardPass(model, input) {
  let activations = input;
  for (const layer of model.layers) {
    activations = denseForward(activations, layer.weights, layer.biases);
    activations = layer.activation === "softmax" ? softmax(activations) : relu(activations);
  }

  let bestIndex = 0;
  for (let i = 1; i < activations.length; i++) {
    if (activations[i] > activations[bestIndex]) bestIndex = i;
  }
  return { label: model.labels[bestIndex], confidence: activations[bestIndex] };
}

// Carrega um modelo exportado a partir de um caminho dentro da extensão.
// Devolve null (sem quebrar) se o arquivo ainda não existir — os modelos são
// gerados pelo treino em Python e não vêm no repositório.
export async function loadModelFile(relativePath) {
  try {
    const response = await fetch(chrome.runtime.getURL(relativePath));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    console.warn(`Modelo não disponível (${relativePath}):`, err.message);
    return null;
  }
}
