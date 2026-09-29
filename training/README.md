# Treino dos classificadores de letras

Scripts Python usados **fora** da extensão, só para gerar os modelos que a
extensão carrega depois. Rodam localmente, uma vez (ou sempre que você quiser
melhorar o dataset).

São **dois** classificadores, porque o alfabeto de Libras tem dois tipos de
letra:

| | Letras | Coleta | Treino | Modelo gerado |
|---|---|---|---|---|
| **Paradas** (pose) | as outras 21 | `collect_data.py` | `train.py` | `letter_classifier.json` |
| **Com movimento** | H, J, K, X, Z | `collect_movement_data.py` | `train_movement.py` | `movement_classifier.json` |

As 5 letras com movimento não são poses: a mão se desloca ou gira. Não dá pra
reconhecê-las olhando uma "foto" por vez — no meio de um J, por exemplo, a mão
passa por formas que parecem outras letras paradas. Por isso elas têm um
classificador próprio, que olha o **trajeto** em vez da pose.

## Passo a passo (letras paradas)

É preciso Python 3.10–3.12 (versões muito novas ainda não têm suporte
completo em algumas bibliotecas usadas aqui). Se `python --version` mostrar
algo mais novo que 3.12, use o Python Launcher do Windows para escolher a
versão certa (`py -3.12 -m venv .venv`).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Coletar amostras: abre a webcam, você faz a pose da letra e pressiona
#    a tecla correspondente (A-Z) para gravar. Repita ~30-50x por letra,
#    variando um pouco o ângulo/posição da mão.
.venv\Scripts\python.exe collect_data.py

# 2. Treinar e exportar o modelo
.venv\Scripts\python.exe train.py
```

O `train.py` treina uma rede neural pequena com scikit-learn (de propósito,
sem TensorFlow — evita um monte de conflito de versão para um modelo tão
simples) e gera `extension/lib/models/letter_classifier.json`, com os pesos
do modelo e a ordem das letras. `extension/src/classifier/classifier.js` já
sabe carregar e aplicar esse arquivo automaticamente — não precisa editar
nada, é só recarregar a extensão no Chrome depois de treinar.

## Passo a passo (letras com movimento: H, J, K, X, Z)

⚠️ **Antes de gravar**: confira numa fonte confiável (dicionário do
[INES](https://www.ines.gov.br/) ou vídeo de sinalizante nativo) como cada uma
dessas 5 letras é feita de verdade. Se gravar o movimento errado, o modelo vai
aprender o movimento errado com muita precisão — o que é pior do que não ter a
letra.

```powershell
# 1. Gravar os gestos. Aperte H, J, K, X ou Z: ele conta 3, 2, 1 e grava o
#    movimento inteiro (1,5s). Aperte ESPAÇO para gravar "NADA" — mexer a mão
#    à toa, levar ela até uma posição. Grave ~40 de cada letra e ~60 de NADA.
.venv\Scripts\python.exe collect_movement_data.py

# 2. Treinar e exportar
.venv\Scripts\python.exe train_movement.py
```

**Por que gravar "NADA"**: levar a mão até a posição também é movimento. Em vez
de tentar adivinhar por regra o que é gesto e o que é só transição, ensinamos o
modelo com exemplos de "isso aqui não é letra nenhuma" e deixamos ele decidir.
Sem essas amostras, qualquer mexida da mão viraria letra.

**Como o gesto vira números**: cada gravação (uma sequência de frames) é
resumida em 60 características por `movement_features.py` — a forma da mão no
início e no fim, o trajeto do pulso, o quão reto foi o caminho (separa o K, que
sobe reto, do Z, que zigue-zagueia), quantas vezes inverteu de direção, e a
duração. Os landmarks são guardados **crus** em
`data/movement_sequences.csv`, então dá pra melhorar a extração de
características depois sem regravar nada.

### O teste de paridade (importante)

`movement_features.py` tem um gêmeo em JavaScript
(`extension/src/classifier/movementFeatures.js`), porque a extensão precisa
calcular exatamente as mesmas 60 características em tempo real. **Se os dois
saírem de sincronia, o modelo passa a errar sem dar erro nenhum** — só piora a
acurácia em silêncio, que é o bug mais difícil de achar deste projeto.

Por isso existe um teste. Rode-o sempre que mexer em qualquer um dos dois:

```powershell
.venv\Scripts\python.exe check_parity.py    # gera o gesto de referência
cd ..\extension
npm test                                     # confere JS x Python
```

## Dataset externo (opcional): Brazilian Sign Language Alphabet Dataset

Além das suas próprias amostras, dá pra usar um dataset acadêmico público de
imagens do alfabeto de Libras — mantido em **arquivo separado**, sem misturar
com `training/data/landmarks.csv`.

**O que é**: [Brazilian Sign Language Alphabet Dataset](https://biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset/)
(Passos, Fernandes e Comunello, Mendeley Data, DOI 10.17632/k4gs3bmx5k.5),
4.411 imagens 200×200. **Atenção ao limite**: as imagens são originalmente de
um dataset de ASL (língua de sinais americana) — os autores selecionaram só
as 15 letras em que afirmam que a forma da mão é igual em Libras (**A, B, C,
D, E, I, L, M, N, O, R, S, U, V, W**). Não inclui as letras com movimento
(H, J, K, X, Z — o próprio alfabeto de Libras "de verdade" exige movimento
nelas, não cabem numa foto parada) nem foi capturado com sinalizantes
brasileiros nativos. Trate como um reforço de robustez, não como fonte de
verdade absoluta.

**Como usar**:

```powershell
# 1. Baixe o zip (link no README do dataset) e extraia de forma que fique:
#    training/data/external/extracted/A/*.jpg (+ .xml), training/data/external/extracted/B/..., etc.

# 2. Extrai os landmarks de cada imagem (mesmo HandLandmarker usado no resto do projeto)
.venv\Scripts\python.exe process_external_dataset.py
# gera training/data/external/landmarks_external.csv — não mexe no seu landmarks.csv

# 3. Treinar incluindo o dataset externo (opcional — por padrão o train.py usa só o seu)
.venv\Scripts\python.exe train.py --with-external
```

Na nossa execução, a taxa de detecção de mão variou por letra — a maioria
ficou acima de 90%, mas **M e N ficaram bem abaixo** (respectivamente ~48% e
~20% das imagens) porque são formas de mão fechada/com dedos sobrepostos,
mais difíceis do MediaPipe detectar com confiança numa imagem estática — vale
não confiar demais nessas duas classes vindas do dataset externo.

**Isso é 100% reversível**: `--with-external` só junta os dois CSVs *na
memória*, no momento do treino — nunca escreve nada de volta em
`landmarks.csv` nem em `landmarks_external.csv`. Pra voltar a treinar só com
as suas amostras, basta rodar `train.py` sem a flag; o modelo antigo em
`extension/lib/models/letter_classifier.json` é sobrescrito a cada treino, o
que também significa que "desfazer" um treino é só rodar de novo com a
combinação que você quer (não existe um estado misturado permanente).

## Amostras confirmadas na extensão (opcional): terceira fonte de dado

A extensão tem um modo de confirmação por sorriso — quando você confirma uma
letra reconhecida, ela vira uma amostra guardada em memória, e o botão
"Baixar amostras confirmadas (CSV)" exporta um arquivo no **mesmo formato**
de `landmarks.csv` (`p0..p62,label`). Isso é interessante porque são
amostras de uso real (não de uma sessão de gravação dedicada), mas também
**sem garantia de qualidade** — se você confirmou sem querer ou o sorriso
disparou errado, a amostra entra do mesmo jeito.

**Como incorporar** (mesmo espírito "separado até você decidir juntar" do
dataset externo acima): salve o CSV baixado em `training/data/`, dê uma
olhada rápida nas linhas antes de confiar nelas, e se quiser usá-lo no
treino, é só concatenar manualmente com `landmarks.csv` (ou, se for usar com
frequência, seguir o mesmo padrão do `--with-external` e adaptar `train.py`
— não fiz isso automaticamente de propósito, pra não presumir que você
sempre vai querer misturar).

## Por que landmarks e não pixels da imagem

O classificador não olha para a imagem crua — ele recebe os 21 pontos 3D da
mão que o MediaPipe já extraiu (o mesmo `HandLandmarker` que a extensão usa
em `app.js`), normalizados (centralizados no pulso, escalados pela maior
distância). Isso deixa o problema muito mais simples (63 números de entrada
em vez de uma imagem inteira), robusto a variação de iluminação/fundo, e é
exatamente o mesmo pré-processamento feito em tempo real pela extensão
(`normalizeLandmarks` em `classifier.js`) — por isso as duas implementações
precisam ficar em sincronia.
