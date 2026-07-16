"use strict";

const elements = {
  imageInput: document.querySelector("#image-input"),
  dropZone: document.querySelector("#drop-zone"),
  emptyUpload: document.querySelector("#empty-upload"),
  imagePreview: document.querySelector("#image-preview"),
  fileMeta: document.querySelector("#file-meta"),
  clearButton: document.querySelector("#clear-button"),
  analyzeButton: document.querySelector("#analyze-button"),
  buttonLabel: document.querySelector("#analyze-button .button-label"),
  status: document.querySelector("#request-status"),
  apiSettings: document.querySelector("#api-settings"),
  endpointInput: document.querySelector("#endpoint-input"),
  applyEndpoint: document.querySelector("#apply-endpoint"),
  connectionBadge: document.querySelector("#connection-badge"),
  resultState: document.querySelector("#result-state"),
  resultEmpty: document.querySelector("#result-empty"),
  resultContent: document.querySelector("#result-content"),
  referenceValues: document.querySelector("#reference-values"),
  typeValues: document.querySelector("#type-values"),
  jsonOutput: document.querySelector("#json-output"),
  rawOutput: document.querySelector("#raw-output"),
  copyJson: document.querySelector("#copy-json"),
};

const canonicalTypes = [
  [/\bmarkush\s+structures?\b/i, "Markush structure"],
  [/\bsubstituents?\b/i, "substituent"],
  [/\bspecific\s+compounds?\b/i, "specific compound"],
];

const state = {
  file: null,
  previewUrl: null,
  endpoint: "",
  result: null,
};

function spaceVariable(name) {
  return window.huggingface?.variables?.[name] || "";
}

function setStatus(message, tone = "neutral") {
  elements.status.textContent = message;
  elements.status.className = `status-message ${tone}`;
}

function setResultState(label, tone = "idle") {
  elements.resultState.textContent = label;
  elements.resultState.className = `result-state ${tone}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** unitIndex;
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function validateEndpoint(value) {
  const trimmed = value.trim();
  if (!trimmed) return "";
  let url;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error("Enter a complete inference endpoint URL.");
  }
  const localHost = ["localhost", "127.0.0.1", "::1"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && localHost)) {
    throw new Error("Use an HTTPS endpoint. HTTP is accepted only for local development.");
  }
  return url.toString();
}

function updateConnection(endpoint, announce = false) {
  state.endpoint = endpoint;
  elements.endpointInput.value = endpoint;
  if (endpoint) {
    elements.connectionBadge.textContent = "Connected";
    elements.connectionBadge.className = "connection-badge connected";
    if (announce) setStatus("Inference endpoint configured. Select an image and run analysis.", "success");
  } else {
    elements.connectionBadge.textContent = "Not configured";
    elements.connectionBadge.className = "connection-badge disconnected";
    if (announce) setStatus("Static preview mode: configure an inference endpoint to analyze images.", "warning");
  }
}

function clearResult() {
  state.result = null;
  elements.resultContent.hidden = true;
  elements.resultEmpty.hidden = false;
  elements.referenceValues.replaceChildren();
  elements.typeValues.replaceChildren();
  elements.jsonOutput.textContent = "";
  elements.rawOutput.textContent = "";
  setResultState("Waiting", "idle");
}

function clearImage() {
  state.file = null;
  elements.imageInput.value = "";
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  elements.imagePreview.removeAttribute("src");
  elements.imagePreview.hidden = true;
  elements.fileMeta.hidden = true;
  elements.emptyUpload.hidden = false;
  elements.dropZone.classList.remove("has-image");
  elements.clearButton.disabled = true;
  elements.analyzeButton.disabled = true;
  clearResult();
  setStatus("Select an image to begin. The file stays in your browser until you run an API request.");
}

function acceptFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Please select a PNG, JPEG, WebP, or GIF image.", "error");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    setStatus("The selected image exceeds the 25 MB browser upload limit.", "error");
    return;
  }

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  elements.imagePreview.src = state.previewUrl;
  elements.imagePreview.hidden = false;
  elements.fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;
  elements.fileMeta.hidden = false;
  elements.emptyUpload.hidden = true;
  elements.dropZone.classList.add("has-image");
  elements.clearButton.disabled = false;
  elements.analyzeButton.disabled = false;
  clearResult();

  if (state.endpoint) {
    setStatus("Image ready. Run analysis to send it to the configured endpoint.", "success");
  } else {
    setStatus("Image preview ready. Configure an inference endpoint before analysis.", "warning");
  }
}

function balancedEnd(text, start, opening, closing) {
  let depth = 0;
  let quote = null;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const character = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (character === "\\") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (character === quote) quote = null;
      continue;
    }
    if (character === '"') {
      quote = character;
      continue;
    }
    if (character === opening) depth += 1;
    if (character === closing) {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return null;
}

function extractLastBoxedBody(text) {
  const matches = [...text.matchAll(/\\boxed\s*/gi)];
  for (let matchIndex = matches.length - 1; matchIndex >= 0; matchIndex -= 1) {
    const match = matches[matchIndex];
    let cursor = match.index + match[0].length;
    while (/\s/.test(text[cursor] || "")) cursor += 1;
    if (text[cursor] === "{") {
      const end = balancedEnd(text, cursor, "{", "}");
      if (end !== null) return text.slice(cursor + 1, end).trim();
    } else if (cursor < text.length) {
      return text.slice(cursor).trim();
    }
  }
  return null;
}

function splitReferenceNames(text) {
  const items = [];
  const depths = { "(": 0, "[": 0, "{": 0 };
  const pairs = { ")": "(", "]": "[", "}": "{" };
  let start = 0;
  let quote = null;
  let escaped = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (character === "\\") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (character === quote) quote = null;
      continue;
    }
    if (character === '"') {
      quote = character;
      continue;
    }
    if (Object.hasOwn(depths, character)) {
      depths[character] += 1;
      continue;
    }
    if (Object.hasOwn(pairs, character)) {
      const opener = pairs[character];
      depths[opener] = Math.max(0, depths[opener] - 1);
      continue;
    }
    const nested = Object.values(depths).some((depth) => depth > 0);
    if (character === "," && !nested && /\s/.test(text[index + 1] || "")) {
      items.push(text.slice(start, index));
      start = index + 1;
    }
  }
  items.push(text.slice(start));
  return items
    .map((item) => item.trim().replace(/^[`"']+|[`"']+$/g, "").trim())
    .filter(Boolean)
    .map((item) => (item.toLowerCase() === "none" ? "None" : item));
}

function parsePair(text) {
  const candidates = [];
  for (let start = 0; start < text.length; start += 1) {
    if (text[start] !== "[") continue;
    const end = balancedEnd(text, start, "[", "]");
    if (end === null) continue;
    let cursor = end + 1;
    while (/\s/.test(text[cursor] || "")) cursor += 1;
    if (text[cursor] !== ":") continue;

    const typeText = text.slice(cursor + 1);
    const structureTypes = canonicalTypes
      .filter(([pattern]) => pattern.test(typeText))
      .map(([, canonical]) => canonical);
    const referenceNames = splitReferenceNames(text.slice(start + 1, end));
    if (structureTypes.length && referenceNames.length) {
      candidates.push({ reference_names: referenceNames, structure_types: structureTypes });
    }
  }
  return candidates.at(-1) || null;
}

function parseModelOutput(value) {
  const normalized = String(value ?? "").replaceAll("{{", "{").replaceAll("}}", "}").trim();
  const boxedBody = extractLastBoxedBody(normalized);
  const parsed = (boxedBody && parsePair(boxedBody)) || parsePair(normalized);
  if (!parsed) throw new Error("No valid '[reference names]: [structure types]' answer was found.");
  return parsed;
}

function asStringArray(value) {
  if (Array.isArray(value)) return value.map(String).map((item) => item.trim()).filter(Boolean);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function rawTextFromPayload(payload) {
  const value = payload?.raw_output ?? payload?.generated_text ?? payload?.output ?? payload?.text ?? "";
  if (typeof value === "string") return value.trim();
  return value ? JSON.stringify(value, null, 2) : "";
}

function normalizeResponse(payload) {
  const candidate = payload?.result && typeof payload.result === "object" ? payload.result : payload;
  let referenceNames = asStringArray(candidate?.reference_names);
  let structureTypes = asStringArray(candidate?.structure_types);
  const rawOutput = rawTextFromPayload(candidate);

  if (!referenceNames.length || !structureTypes.length) {
    if (!rawOutput) throw new Error("The API response contains neither parsed fields nor raw model output.");
    const parsed = parseModelOutput(rawOutput);
    referenceNames = parsed.reference_names;
    structureTypes = parsed.structure_types;
  }

  return {
    reference_names: referenceNames,
    structure_types: structureTypes,
    parsed: true,
    raw_output: rawOutput,
  };
}

function chip(value) {
  const element = document.createElement("span");
  element.className = "value-chip";
  element.textContent = value;
  return element;
}

function renderResult(result) {
  state.result = result;
  elements.referenceValues.replaceChildren(...result.reference_names.map(chip));
  elements.typeValues.replaceChildren(...result.structure_types.map(chip));
  elements.jsonOutput.textContent = JSON.stringify({
    reference_names: result.reference_names,
    structure_types: result.structure_types,
    parsed: true,
  }, null, 2);
  elements.rawOutput.textContent = result.raw_output || "The endpoint returned parsed fields without raw output.";
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  setResultState("Parsed", "success");
}

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw_output: text };
  }
}

async function analyze() {
  if (!state.file) return;
  if (!state.endpoint) {
    elements.apiSettings.open = true;
    elements.endpointInput.focus();
    setStatus("This is a Static Space. Configure an external inference endpoint to run analysis.", "warning");
    return;
  }

  elements.analyzeButton.disabled = true;
  elements.analyzeButton.classList.add("loading");
  elements.buttonLabel.textContent = "Analyzing…";
  setResultState("Running", "loading");
  setStatus("Uploading the selected image to the configured inference endpoint…", "neutral");

  try {
    const formData = new FormData();
    formData.append("image", state.file, state.file.name);
    const response = await fetch(state.endpoint, { method: "POST", body: formData });
    const payload = await readResponse(response);
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || payload?.message || `HTTP ${response.status}`;
      throw new Error(String(detail));
    }
    renderResult(normalizeResponse(payload));
    setStatus("Analysis complete. The response was parsed into developer-ready fields.", "success");
  } catch (error) {
    clearResult();
    setResultState("Error", "error");
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Inference failed: ${message}`, "error");
  } finally {
    elements.analyzeButton.disabled = !state.file;
    elements.analyzeButton.classList.remove("loading");
    elements.buttonLabel.textContent = "Analyze structure";
  }
}

elements.imageInput.addEventListener("change", () => acceptFile(elements.imageInput.files?.[0]));
elements.clearButton.addEventListener("click", clearImage);
elements.analyzeButton.addEventListener("click", analyze);

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragover");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragover");
  });
}

elements.dropZone.addEventListener("drop", (event) => acceptFile(event.dataTransfer?.files?.[0]));

elements.applyEndpoint.addEventListener("click", () => {
  try {
    updateConnection(validateEndpoint(elements.endpointInput.value), true);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message, "error");
  }
});

elements.endpointInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") elements.applyEndpoint.click();
});

elements.copyJson.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(elements.jsonOutput.textContent);
    elements.copyJson.textContent = "Copied";
    window.setTimeout(() => { elements.copyJson.textContent = "Copy"; }, 1200);
  } catch {
    setStatus("Clipboard access is unavailable in this browser.", "warning");
  }
});

clearImage();
try {
  updateConnection(validateEndpoint(spaceVariable("INFERENCE_API_URL")));
} catch (error) {
  updateConnection("");
  elements.apiSettings.open = true;
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`The INFERENCE_API_URL Space variable is invalid: ${message}`, "error");
}

window.HiroChemicalInsights = Object.freeze({ parseModelOutput, normalizeResponse });
