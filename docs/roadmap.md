# Roadmap — Tradutor de Libras

## Fase 1 — MVP: alfabeto, só extensão (atual)

Extensão de navegador (Manifest V3), 100% client-side:

- `extension/src/app/` — captura de webcam + `HandLandmarker` do MediaPipe
  (`@mediapipe/tasks-vision`, rodando via WASM local, sem CDN) desenhando os
  21 landmarks da mão em tempo real.
- `extension/src/classifier/` — carrega `extension/lib/models/letter_classifier.json`
  (pesos de uma rede neural pequena treinada com scikit-learn) e aplica um
  forward pass manual em JavaScript puro — sem TensorFlow.js, de propósito
  (evita conflito de versão pra um modelo tão simples).
- `extension/src/app/app.js` só mostra uma letra quando o classificador tem
  pelo menos 80% de confiança (`CONFIDENCE_THRESHOLD`); abaixo disso mostra
  "—", mesmo com a mão detectada na tela — evita mostrar letra errada só
  porque a mão apareceu no quadro.
- Sem backend, sem autenticação, sem Java nesta fase.

**Status atual**: alfabeto completo, com **dois classificadores** — porque o
alfabeto de Libras tem dois tipos de letra:

- **21 letras paradas** (pose): `classifier.js` + `letter_classifier.json`,
  treinado por `training/train.py` a partir de `collect_data.py`. Dá pra
  reforçar com um dataset público (`process_external_dataset.py` +
  `train.py --with-external`), mantido em arquivo separado das suas amostras.
- **5 letras com movimento** (H, J, K, X, Z): `movementClassifier.js` +
  `movement_classifier.json`, treinado por `training/train_movement.py` a
  partir de `collect_movement_data.py`. Essas letras não são poses — a mão se
  desloca ou gira —, então são classificadas pelo **trajeto**, não pela forma
  num instante.

**Como os dois convivem** (`extension/src/app/gestureRecorder.js`): a cada
quadro, mede-se o maior deslocamento entre o pulso e as pontas dos dedos, em
"tamanhos de mão por segundo" (assim o limiar vale igual com a mão perto ou
longe da câmera). Medir só o pulso não funcionava: no H quem se mexe são os
dedos, e o pulso fica praticamente parado.

```
mão parada   -> classificador estático (a pose é a letra)
mão mexendo  -> grava a sequência até parar -> classificador de movimento
```

Sem essa arbitragem, o classificador estático ficaria gritando letras erradas
no meio de um gesto: durante um J a mão passa por formas que parecem outras
letras paradas — foi exatamente isso que aconteceu quando o K estava na base
estática e derrubava a acurácia do resto.

O classificador de movimento tem uma sexta classe, **"NADA"**, treinada com
gravações de mão se mexendo à toa. É ela que evita que simplesmente levar a
mão até a posição vire uma letra: em vez de adivinhar por regra o que é gesto
e o que é transição, o próprio modelo decide.

Depois de reconhecer uma letra de movimento, ela fica na tela por
`MOVEMENT_HOLD_MS` (~1,8s) — senão o classificador estático assumiria de volta
no quadro seguinte (a mão continua parada em alguma pose no fim do gesto) e
não daria tempo de sorrir pra confirmar.

**As duas implementações que precisam ficar em sincronia**: as 60
características do gesto são calculadas em Python (`movement_features.py`,
para treinar) e em JavaScript (`movementFeatures.js`, em tempo real). Se
divergirem, o modelo erra **sem dar erro nenhum**. Existe um teste pra isso —
`training/check_parity.py` + `npm test` na pasta `extension/`.

**Confirmação por sorriso**: a letra reconhecida na tela é só "tentativa" —
ela só é falada em voz alta (Web Speech API do navegador, `pt-BR`, sem
servidor nenhum) e salva como amostra depois que você **sorri** pra
confirmar. Detecção em `extension/src/app/smileDetector.js`: usa os
blendshapes do `FaceLandmarker` (outro modelo do MediaPipe, além do de
mãos, com `outputFaceBlendshapes: true`) — o próprio MediaPipe já entrega um
número de 0 a 1 pra "quanto está sorrindo" (`mouthSmileLeft`/`mouthSmileRight`),
sem precisar de nenhuma conta de geometria. Escolhido no lugar de aceno de
cabeça (testado antes, descartado): sorrir não mexe a mão que está fazendo a
letra, então não atrapalha o reconhecimento como o aceno atrapalhava. Ao
confirmar: toca um bipe sintetizado (`sound.js`), fala a letra, e guarda a
amostra em memória (`sampleStore.js`) — um botão na extensão baixa essas
amostras confirmadas como CSV, no mesmo formato de
`training/data/landmarks.csv`, pra você incorporar no treino manualmente
depois (terceira fonte de dado, junto de `landmarks.csv` e
`landmarks_external.csv` — mesmo espírito "combina só quando eu pedir", ver
`training/README.md`). Modelo de rosto é opcional: sem ele baixado, a
extensão reconhece letras normalmente, só não confirma. Limiares de
score/tempo do sorriso são um chute inicial, precisam de ajuste ao vivo
testando com câmera de verdade.

Só letras **paradas** viram amostra confirmada: o CSV tem o formato de uma
pose única (63 números de um quadro só), que não representa um gesto — salvar
uma letra de movimento ali colocaria uma foto solta do meio do movimento no
dataset estático.

**Limiares que precisam de ajuste ao vivo**: os números do
`gestureRecorder.js` (velocidade pra considerar que começou/parou um gesto,
duração mínima) e do `smileDetector.js` são chutes iniciais — não dá pra
calibrá-los sem câmera e mão de verdade. Estão todos como constantes
comentadas no topo de cada arquivo.

## Fase 2 — Frases, tradução e áudio (entra o Java)

Quando o reconhecimento de letra isolada estiver estável, o próximo passo é
reconhecer *sequências* (palavras/frases), traduzir a gramática de Libras
para português e gerar áudio. Isso deixa de ser algo que dá para fazer só no
navegador:

- `orchestrator-service/` (**Java + Spring Boot**): recebe da extensão a
  sequência de sinais reconhecidos (via REST ou WebSocket), orquestra as
  chamadas para os serviços abaixo, e devolve o resultado final (texto +
  áudio) para a extensão.
- `translation-service/`: aplica a estrutura gramatical de Libras → Português
  (Libras não é "português sinalizado" — tem gramática própria, então isso é
  um passo de NLP, não tradução palavra-por-palavra), e depois tradução para
  outros idiomas.
- `tts-service/`: converte o texto final em áudio.
- Extensão passa a ter um `background`/módulo que fala com o
  `orchestrator-service` quando o usuário terminar uma frase.

## Fase 3 — Expansão

- Modelo temporal (LSTM/Transformer sobre a sequência de landmarks ao longo
  do tempo) para reconhecer sinais de palavras inteiras, não só letra por
  letra soletrada.
- Mais idiomas de saída.
- Reavaliar se ainda faz sentido ser "extensão de navegador" ou se compensa
  virar um app dedicado, dependendo de como o uso evoluir.
