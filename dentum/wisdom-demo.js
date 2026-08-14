const defaultImage = "./assets/wisdom-tooth-removal/panoramic-demo.jpg";
const modelPath = "./assets/models/dental-panoramic-yolo11n.onnx";
const modelSize = 640;
const confidenceThreshold = 0.45;

let modelSessionPromise = null;

const state = {
  hasImage: true,
  imageUrl: defaultImage,
  tooth: null,
  guess: null,
  resultVisible: false,
  analyzing: false,
};

const uploadInput = document.querySelector("#xray-upload");
const useExampleButton = document.querySelector("#use-example");
const uploadMessage = document.querySelector("#upload-message");
const image = document.querySelector("#xray-image");
const fileName = document.querySelector("#file-name");
const hint = document.querySelector("#xray-hint");
const showResultButton = document.querySelector("#show-result");
const resultPanel = document.querySelector("#result-panel");
const emptyResult = document.querySelector("#empty-result");
const resultTitle = document.querySelector("#result-title");
const matchBadge = document.querySelector("#match-badge");
const resetButton = document.querySelector("#reset-demo");
const tryAgainButton = document.querySelector("#try-again");
const consultationButton = document.querySelector("#book-consultation");
const consultationDialog = document.querySelector("#consultation-dialog");
const xrayStage = document.querySelector("#xray-stage");
const modelStatus = document.querySelector("#model-status");
const reasonList = document.querySelector("#result-reason-list");

const toothButtons = [...document.querySelectorAll("[data-tooth-choice], [data-tooth-marker]")];
const guessButtons = [...document.querySelectorAll("[data-guess]")];
const steps = [...document.querySelectorAll(".step")];

function setStepStates() {
  const uploadStep = steps.find((step) => step.dataset.step === "upload");
  const toothStep = steps.find((step) => step.dataset.step === "tooth");
  const guessStep = steps.find((step) => step.dataset.step === "guess");

  uploadStep.classList.toggle("is-complete", state.hasImage);
  uploadStep.classList.toggle("is-active", !state.hasImage);
  toothStep.classList.toggle("is-active", state.hasImage && !state.tooth);
  toothStep.classList.toggle("is-complete", Boolean(state.tooth));
  guessStep.classList.toggle("is-active", Boolean(state.tooth));
  guessStep.classList.toggle("is-complete", Boolean(state.guess));
}

function updateControls() {
  toothButtons.forEach((button) => {
    const selected = (button.dataset.toothChoice || button.dataset.toothMarker) === state.tooth;
    button.setAttribute("aria-pressed", String(selected));
  });

  guessButtons.forEach((button) => {
    button.setAttribute("aria-checked", String(button.dataset.guess === state.guess));
  });

  showResultButton.disabled = state.analyzing || !(state.hasImage && state.tooth && state.guess);
  hint.textContent = state.tooth ? `Выбран зуб ${state.tooth}` : "Выберите 38 или 48 на снимке";
  setStepStates();
}

function selectTooth(tooth) {
  state.tooth = tooth;
  state.resultVisible = false;
  resultPanel.hidden = true;
  emptyResult.hidden = false;
  updateControls();
}

function selectGuess(guess) {
  state.guess = guess;
  state.resultVisible = false;
  resultPanel.hidden = true;
  emptyResult.hidden = false;
  updateControls();
}

function loadExample() {
  if (state.imageUrl && state.imageUrl.startsWith("blob:")) {
    URL.revokeObjectURL(state.imageUrl);
  }
  state.hasImage = true;
  state.imageUrl = defaultImage;
  image.src = defaultImage;
  image.alt = "Демонстрационный панорамный рентгеновский снимок с нижними зубами мудрости";
  fileName.textContent = "Демонстрационный пример";
  uploadMessage.textContent = "";
  updateControls();
}

function resetDemo({ keepImage = false } = {}) {
  state.tooth = null;
  state.guess = null;
  state.resultVisible = false;
  resultPanel.hidden = true;
  emptyResult.hidden = false;
  if (!keepImage) loadExample();
  updateControls();
}

function handleFile(file) {
  uploadMessage.textContent = "";
  const supportedTypes = ["image/jpeg", "image/png", "image/webp"];
  const maxSize = 15 * 1024 * 1024;

  if (!supportedTypes.includes(file.type)) {
    uploadMessage.textContent = "Нужен файл JPG, PNG или WEBP.";
    uploadInput.value = "";
    return;
  }

  if (file.size > maxSize) {
    uploadMessage.textContent = "Файл больше 15 МБ. Выберите уменьшенную копию.";
    uploadInput.value = "";
    return;
  }

  if (state.imageUrl && state.imageUrl.startsWith("blob:")) {
    URL.revokeObjectURL(state.imageUrl);
  }

  state.imageUrl = URL.createObjectURL(file);
  state.hasImage = true;
  image.src = state.imageUrl;
  image.alt = "Загруженный панорамный снимок";
  fileName.textContent = file.name;
  resetDemo({ keepImage: true });
}

function waitForImage() {
  if (image.complete && image.naturalWidth) return Promise.resolve();
  return new Promise((resolve, reject) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", () => reject(new Error("Не удалось прочитать снимок.")), { once: true });
  });
}

function prepareInputTensor() {
  const canvas = document.createElement("canvas");
  canvas.width = modelSize;
  canvas.height = modelSize;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.min(modelSize / image.naturalWidth, modelSize / image.naturalHeight);
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  const offsetX = (modelSize - width) / 2;
  const offsetY = (modelSize - height) / 2;

  context.fillStyle = "#000";
  context.fillRect(0, 0, modelSize, modelSize);
  context.drawImage(image, offsetX, offsetY, width, height);

  const pixels = context.getImageData(0, 0, modelSize, modelSize).data;
  const planeSize = modelSize * modelSize;
  const input = new Float32Array(planeSize * 3);
  for (let index = 0; index < planeSize; index += 1) {
    input[index] = pixels[index * 4] / 255;
    input[planeSize + index] = pixels[index * 4 + 1] / 255;
    input[planeSize * 2 + index] = pixels[index * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", input, [1, 3, modelSize, modelSize]);
}

async function getModelSession() {
  if (!window.ort) throw new Error("Библиотека модели не загрузилась.");
  if (!modelSessionPromise) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    modelSessionPromise = ort.InferenceSession.create(modelPath, { executionProviders: ["wasm"] });
  }
  return modelSessionPromise;
}

function findSelectedTooth(output) {
  const [, first, second] = output.dims;
  const attributesFirst = first <= second;
  const attributeCount = attributesFirst ? first : second;
  const detectionCount = attributesFirst ? second : first;
  if (attributeCount < 7) throw new Error("Неожиданный формат ответа модели.");

  const valueAt = (attribute, detection) => (
    attributesFirst
      ? output.data[attribute * detectionCount + detection]
      : output.data[detection * attributeCount + attribute]
  );

  let best = null;
  for (let detection = 0; detection < detectionCount; detection += 1) {
    const confidence = valueAt(6, detection);
    if (confidence < confidenceThreshold) continue;
    const centerX = valueAt(0, detection);
    const centerY = valueAt(1, detection);
    const isLowerJaw = centerY > modelSize * 0.5;
    const isSelectedSide = state.tooth === "48" ? centerX < modelSize * 0.5 : centerX >= modelSize * 0.5;
    if (!isLowerJaw || !isSelectedSide) continue;
    if (!best || confidence > best.confidence) best = { confidence, centerX, centerY };
  }
  return best;
}

function makeDistribution(detection) {
  if (!detection) return { simple: 40, medium: 38, complex: 22 };
  const complex = Math.round(25 + detection.confidence * 55);
  const medium = Math.round(45 - detection.confidence * 25);
  return { simple: 100 - complex - medium, medium, complex };
}

function renderDistribution(distribution) {
  Object.entries(distribution).forEach(([level, value]) => {
    const row = document.querySelector(`.probability[data-level="${level}"]`);
    row.querySelector("i").style.setProperty("--value", `${value}%`);
    row.querySelector("strong").textContent = `${value}%`;
  });
}

function renderReasons(detection) {
  const reasons = detection
    ? [
        `Детектор нашёл ретинированный зуб: уверенность ${Math.round(detection.confidence * 100)}%`,
        `Объект найден в зоне зуба ${state.tooth}`,
        "Сложность рассчитана игровой эвристикой поверх детекции",
      ]
    : [
        `В зоне зуба ${state.tooth} нет детекции выше ${Math.round(confidenceThreshold * 100)}%`,
        "Показано нейтральное игровое распределение",
        "Нужна проверка снимка хирургом",
      ];
  reasonList.replaceChildren(...reasons.map((reason) => {
    const item = document.createElement("li");
    const marker = document.createElement("span");
    marker.setAttribute("aria-hidden", "true");
    item.append(marker, document.createTextNode(reason));
    return item;
  }));
}

async function showResult() {
  if (showResultButton.disabled) return;
  state.analyzing = true;
  showResultButton.textContent = "Анализируем…";
  modelStatus.textContent = "Загрузка модели 10 МБ и анализ снимка…";
  updateControls();

  try {
    await waitForImage();
    const session = await getModelSession();
    const tensor = prepareInputTensor();
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const detection = findSelectedTooth(outputs[session.outputNames[0]]);
    const distribution = makeDistribution(detection);
    const predicted = Object.entries(distribution).sort((a, b) => b[1] - a[1])[0][0];
    const labels = { simple: "простое", medium: "среднее", complex: "сложное" };

    state.resultVisible = true;
    emptyResult.hidden = true;
    resultPanel.hidden = false;
    resultTitle.textContent = detection
      ? `Зуб ${state.tooth}: игровая оценка — ${labels[predicted]} удаление`
      : `Зуб ${state.tooth}: модель не дала уверенной детекции`;
    renderDistribution(distribution);
    renderReasons(detection);

    const matched = state.guess === predicted;
    matchBadge.textContent = matched ? "Ваш прогноз совпал" : "Система оценила иначе";
    matchBadge.classList.toggle("is-miss", !matched);
    modelStatus.textContent = detection
      ? `Модель сработала: уверенность детекции ${Math.round(detection.confidence * 100)}%.`
      : "Модель сработала, но не нашла выбранный зуб с достаточной уверенностью.";
    resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    modelSessionPromise = null;
    modelStatus.textContent = `Ошибка модели: ${error.message}`;
  } finally {
    state.analyzing = false;
    showResultButton.textContent = "Запустить модель";
    updateControls();
  }
}

toothButtons.forEach((button) => {
  button.addEventListener("click", () => selectTooth(button.dataset.toothChoice || button.dataset.toothMarker));
});

guessButtons.forEach((button) => {
  button.addEventListener("click", () => selectGuess(button.dataset.guess));
});

uploadInput.addEventListener("change", () => {
  const [file] = uploadInput.files;
  if (file) handleFile(file);
});

useExampleButton.addEventListener("click", () => {
  loadExample();
  resetDemo({ keepImage: true });
});

xrayStage.addEventListener("dragover", (event) => {
  event.preventDefault();
  xrayStage.classList.add("is-dragging");
});

xrayStage.addEventListener("dragleave", () => xrayStage.classList.remove("is-dragging"));

xrayStage.addEventListener("drop", (event) => {
  event.preventDefault();
  xrayStage.classList.remove("is-dragging");
  const [file] = event.dataTransfer.files;
  if (file) handleFile(file);
});

showResultButton.addEventListener("click", () => showResult());
resetButton.addEventListener("click", () => resetDemo());
tryAgainButton.addEventListener("click", () => resetDemo({ keepImage: true }));
consultationButton.addEventListener("click", () => consultationDialog.showModal());

updateControls();

if (new URLSearchParams(window.location.search).get("result") === "1") {
  selectTooth("48");
  selectGuess("complex");
  showResult();
}
