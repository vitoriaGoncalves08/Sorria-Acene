// Copia os arquivos necessários de node_modules para dentro da extensão,
// já que o CSP do Manifest V3 não permite carregar scripts/wasm de CDN.
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const nodeModules = path.join(root, "node_modules");
const lib = path.join(root, "lib");

function copy(src, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
  console.log("copiado:", path.relative(root, dest));
}

// MediaPipe Tasks Vision (bundle ESM + wasm)
const mpSrc = path.join(nodeModules, "@mediapipe", "tasks-vision");
copy(path.join(mpSrc, "vision_bundle.mjs"), path.join(lib, "mediapipe", "vision_bundle.mjs"));
for (const file of fs.readdirSync(path.join(mpSrc, "wasm"))) {
  copy(path.join(mpSrc, "wasm", file), path.join(lib, "mediapipe", "wasm", file));
}

console.log("\nVendoring concluído. Falta baixar o modelo hand_landmarker.task (ver README.md).");
