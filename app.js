// app.js - PostureGuard AI (educational prototype)
// Camera + MediaPipe Pose Landmarker + UI. All processing is local; nothing is uploaded or stored.

import { computeMeasurements, IDX } from "./angles.js";
import { Session } from "./session.js";

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */
const MP_VERSION = "0.10.14"; // pinned MediaPipe Tasks Vision version
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const REMOTE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// Optional offline mode: open the page with ?local=1 to use files from ./vendor (see README).
const USE_LOCAL = new URLSearchParams(location.search).has("local");
const BUNDLE_URL = USE_LOCAL ? "./vendor/vision_bundle.mjs" : `${CDN}/vision_bundle.mjs`;
const WASM_DIR = USE_LOCAL ? "./vendor/wasm" : `${CDN}/wasm`;
const MODEL_URL = USE_LOCAL ? "./vendor/pose_landmarker_lite.task" : REMOTE_MODEL;

const MODEL_TIMEOUT_MS = 30000;
const DRAW_MIN_VIS = 0.5;

// Standard 33-point pose skeleton (index pairs).
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];

const COLORS = { teal: "#14b8a6", blue: "#38bdf8", red: "#ef4444", white: "#ffffff" };

/* ------------------------------------------------------------------ */
/* State (kept in memory only)                                         */
/* ------------------------------------------------------------------ */
const settings = { threshold: 35, duration: 5, sound: false, facing: "environment" };

let state = "idle"; // idle | starting | running | switching
let landmarker = null;
let modelPromise = null;
let delegateUsed = "";
let forceCpu = false;
let stream = null;
let session = null;
let startToken = 0;
let loopId = 0;
let rafId = 0;
let lastVideoTime = -1;
let fpsEma = 0;
let lastInferMs = 0;
let frameCounter = 0;
let inferenceErrors = 0;
let lastSummary = null;
let audioCtx = null;

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
function show(name) {
  for (const id of ["home", "live", "summary"]) $(id).hidden = id !== name;
  document.body.dataset.screen = name;
  window.scrollTo(0, 0);
}

function fmtClock(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
const fmtDeg = (v) => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(1)}°`);
const fmtSec = (v) => `${v.toFixed(1)} s`;

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */
function showHomeError(title, body, { retryModel = false } = {}) {
  $("errorTitle").textContent = title;
  $("errorBody").textContent = body;
  $("retryModelBtn").hidden = !retryModel;
  $("errorCard").hidden = false;
}
function clearHomeError() {
  $("errorCard").hidden = true;
}

function describeCameraError(e) {
  const name = e && e.name;
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      title: "Camera access was blocked",
      body:
        "PostureGuard needs the camera to analyse posture. In Safari, tap the “aA” icon in the address bar → Website Settings → Camera → Allow, then reload this page and tap Start Monitoring again. " +
        "If it still fails, go to Settings → Safari → Camera and choose Ask or Allow.",
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return {
      title: "No camera found",
      body: "This device did not provide the selected camera. Open Settings and try the other camera (front or rear).",
    };
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return {
      title: "The camera is busy",
      body: "Another app or browser tab may be using the camera. Close other apps that use the camera, then try again.",
    };
  }
  return {
    title: "The camera could not start",
    body: `${(e && e.message) || "Unknown error"}. Reload the page and try again.`,
  };
}

function describeModelError(e) {
  if (e && e.message === "TIMEOUT") {
    return {
      title: "The AI model took too long to load",
      body:
        `Loading did not finish within ${MODEL_TIMEOUT_MS / 1000} seconds. Check the internet connection (the model files are downloaded once from jsDelivr and Google), then tap Retry. ` +
        "If the venue Wi-Fi blocks these sites, use the offline option described in the README (?local=1).",
    };
  }
  return {
    title: "The AI model could not be loaded",
    body: `${(e && e.message) || "Unknown error"}. Check the internet connection and tap Retry. If it keeps failing, use the offline option in the README (?local=1).`,
  };
}

function failToHome(info, opts) {
  state = "idle";
  stopLoop();
  stopCamera();
  show("home");
  showHomeError(info.title, info.body, opts);
}

/* ------------------------------------------------------------------ */
/* Model loading (with timeout, GPU -> CPU fallback)                   */
/* ------------------------------------------------------------------ */
function setModelPill(kind, text) {
  const pill = $("modelPill");
  pill.classList.remove("loading", "ready", "error");
  if (kind) pill.classList.add(kind);
  $("modelPillText").textContent = text;
}

async function createLandmarker() {
  const mod = await import(BUNDLE_URL);
  const { PoseLandmarker, FilesetResolver } = mod;
  const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
  const common = {
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  if (!forceCpu) {
    try {
      const lm = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        ...common,
      });
      delegateUsed = "GPU";
      return lm;
    } catch (gpuErr) {
      console.warn("GPU delegate unavailable, falling back to CPU:", gpuErr);
    }
  }
  const lm = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    ...common,
  });
  delegateUsed = "CPU";
  return lm;
}

function loadModel() {
  if (landmarker) return Promise.resolve(landmarker);
  if (modelPromise) return modelPromise;

  setModelPill("loading", "AI model: loading…");
  let timedOut = false;
  let timer = 0;

  const work = createLandmarker().then((lm) => {
    if (timedOut) {
      try { lm.close(); } catch (_) { /* ignore */ }
    }
    return lm;
  });
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      timedOut = true;
      reject(new Error("TIMEOUT"));
    }, MODEL_TIMEOUT_MS);
  });

  modelPromise = Promise.race([work, timeout])
    .then((lm) => {
      clearTimeout(timer);
      landmarker = lm;
      setModelPill("ready", `AI model: ready (${delegateUsed})`);
      return lm;
    })
    .catch((err) => {
      clearTimeout(timer);
      modelPromise = null;
      setModelPill("error", "AI model: failed to load");
      throw err;
    });
  // Avoid "unhandled rejection" noise when nobody is awaiting yet (page preload).
  modelPromise.catch(() => {});
  return modelPromise;
}

function disposeModel() {
  try { if (landmarker) landmarker.close(); } catch (_) { /* ignore */ }
  landmarker = null;
  modelPromise = null;
}

/* ------------------------------------------------------------------ */
/* Camera                                                              */
/* ------------------------------------------------------------------ */
function waitForVideoSize(timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    if (video.videoWidth > 0 && video.readyState >= 2) return resolve();
    const t = setTimeout(() => cleanup(new Error("The camera did not deliver any video.")), timeoutMs);
    const onReady = () => { if (video.videoWidth > 0) cleanup(); };
    function cleanup(err) {
      clearTimeout(t);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("playing", onReady);
      err ? reject(err) : resolve();
    }
    video.addEventListener("loadeddata", onReady);
    video.addEventListener("playing", onReady);
  });
}

function onTrackEnded() {
  if (state === "running") {
    stopMonitoring("The camera stopped unexpectedly, so the session was ended.");
  }
}

async function openCamera() {
  stopCamera();
  const s = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: settings.facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  stream = s;
  s.getVideoTracks().forEach((t) => t.addEventListener("ended", onTrackEnded));
  video.muted = true;
  video.setAttribute("playsinline", "");
  video.srcObject = s;
  await video.play();
  await waitForVideoSize();
  updateStageAspect();
  $("stage").classList.toggle("mirror", settings.facing === "user");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => {
      t.removeEventListener("ended", onTrackEnded);
      t.stop();
    });
    stream = null;
  }
  try { video.pause(); } catch (_) { /* ignore */ }
  video.srcObject = null;
}

function updateStageAspect() {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    $("stage").style.setProperty("--ar", String(video.videoWidth / video.videoHeight));
  }
}
video.addEventListener("resize", updateStageAspect);

/* ------------------------------------------------------------------ */
/* Sound (optional, off by default)                                    */
/* ------------------------------------------------------------------ */
function unlockAudio() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
  } catch (_) { /* ignore */ }
}

function beep() {
  if (!settings.sound || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.4, audioCtx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.6);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.65);
  } catch (_) { /* ignore */ }
}

/* ------------------------------------------------------------------ */
/* Start / stop / reset                                                */
/* ------------------------------------------------------------------ */
function supportError() {
  if (!window.isSecureContext) {
    return {
      title: "This page must be opened over HTTPS",
      body: "Safari only allows camera access on secure (https://) pages. Open the HTTPS link you got when you deployed the app (see README).",
    };
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return { title: "Camera not supported", body: "This browser does not support camera access. Please use Safari on an iPad (iPadOS 15 or newer)." };
  }
  if (typeof WebAssembly === "undefined") {
    return { title: "Browser not supported", body: "This browser cannot run the AI model (WebAssembly missing). Please update iPadOS / Safari." };
  }
  return null;
}

async function startMonitoring() {
  if (state !== "idle") return;
  clearHomeError();
  const unsupported = supportError();
  if (unsupported) return failToHome(unsupported);

  const token = ++startToken;
  state = "starting";
  unlockAudio();
  resetLiveUI();
  show("live");
  $("stageLoading").hidden = false;
  $("stageLoadingText").textContent = "Starting camera and AI model…";

  // Camera is requested directly from the tap (required by iOS), model loads in parallel.
  const camPromise = openCamera();
  const [camRes, modelRes] = await Promise.allSettled([camPromise, loadModel()]);

  if (token !== startToken) {
    // Cancelled while starting.
    stopCamera();
    return;
  }
  if (camRes.status === "rejected") {
    console.error(camRes.reason);
    return failToHome(describeCameraError(camRes.reason));
  }
  if (modelRes.status === "rejected") {
    console.error(modelRes.reason);
    return failToHome(describeModelError(modelRes.reason), { retryModel: true });
  }

  session = new Session(settings);
  session.start(performance.now());
  lastVideoTime = -1;
  fpsEma = 0;
  lastInferMs = 0;
  frameCounter = 0;
  inferenceErrors = 0;
  state = "running";
  $("stageLoading").hidden = true;
  setControls();
  startLoop();
}

function cancelStart() {
  if (state !== "starting") return;
  startToken++;
  stopCamera();
  state = "idle";
  show("home");
}

function stopMonitoring(note = "") {
  if (state !== "running") return;
  stopLoop();
  session.stop(performance.now());
  state = "idle";
  stopCamera();
  lastSummary = session.summary();
  renderSummary(lastSummary, note);
  show("summary");
  requestAnimationFrame(() => drawChart(lastSummary));
}

function resetSession() {
  if (state !== "running") return;
  session.start(performance.now());
  $("alertLog").innerHTML = '<li class="empty">No alerts yet.</li>';
  $("alertBanner").hidden = true;
  $("stage").classList.remove("alert");
}

async function switchCamera() {
  const next = settings.facing === "user" ? "environment" : "user";
  setFacing(next);
  if (state !== "running") return;

  state = "switching";
  stopLoop();
  $("stageLoading").hidden = false;
  $("stageLoadingText").textContent = "Switching camera…";
  try {
    await openCamera();
  } catch (e) {
    console.error(e);
    // Session data so far is kept; show the summary with a note.
    state = "running";
    stopMonitoring(`Switching camera failed (${describeCameraError(e).title}). The session was ended.`);
    return;
  }
  session.markGap();
  lastVideoTime = -1;
  state = "running";
  $("stageLoading").hidden = true;
  startLoop();
}

function setFacing(value) {
  settings.facing = value;
  document.querySelectorAll('input[name="camera"]').forEach((r) => { r.checked = r.value === value; });
}

/* ------------------------------------------------------------------ */
/* Inference loop (exactly one at a time)                              */
/* ------------------------------------------------------------------ */
function stopLoop() {
  loopId++;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
}

function startLoop() {
  stopLoop();
  const myLoop = loopId;
  const tick = () => {
    if (myLoop !== loopId || state !== "running") return;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = performance.now();
      let result = null;
      try {
        result = landmarker.detectForVideo(video, now);
        inferenceErrors = 0;
      } catch (e) {
        handleInferenceError(e);
        return;
      }
      processResult(result, now);
    }
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}

async function handleInferenceError(e) {
  console.error("Inference error:", e);
  stopLoop();
  inferenceErrors++;
  if (delegateUsed === "GPU" && !forceCpu) {
    // GPU can fail at first use on some devices: rebuild on CPU and continue.
    forceCpu = true;
    state = "switching";
    $("stageLoading").hidden = false;
    $("stageLoadingText").textContent = "Switching AI to CPU mode…";
    disposeModel();
    try {
      await loadModel();
      session.markGap();
      state = "running";
      $("stageLoading").hidden = true;
      startLoop();
      return;
    } catch (err) {
      console.error(err);
    }
  }
  state = "running";
  stopMonitoring("The AI model stopped working during the session, so it was ended.");
}

function processResult(result, now) {
  const lm = result && result.landmarks && result.landmarks[0];
  const aspect = video.videoWidth / video.videoHeight;
  const m = lm ? computeMeasurements(lm, aspect) : { valid: false, reason: "No person detected" };
  const out = session.frame(m, now);

  drawOverlay(lm, m, out);
  updateReadouts(out, m);

  if (out.alertTriggered) {
    const a = session.alerts[session.alerts.length - 1];
    addAlertToLog(a);
    beep();
  }

  // FPS (smoothed)
  if (lastInferMs) {
    const inst = 1000 / Math.max(1, now - lastInferMs);
    fpsEma = fpsEma ? 0.9 * fpsEma + 0.1 * inst : inst;
  }
  lastInferMs = now;
  if (++frameCounter % 15 === 0) {
    $("activeText").textContent = `AI processing active · ${fpsEma.toFixed(0)} fps · ${delegateUsed}`;
  }
}

/* ------------------------------------------------------------------ */
/* Drawing                                                             */
/* ------------------------------------------------------------------ */
function drawOverlay(lm, m, out) {
  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
  const W = overlay.width;
  const H = overlay.height;
  octx.clearRect(0, 0, W, H);
  if (!lm) return;

  const unit = Math.max(2, W / 320);
  const visible = (p) => p && (p.visibility ?? 0) >= DRAW_MIN_VIS;
  const px = (p) => [p.x * W, p.y * H];

  // Skeleton
  octx.lineCap = "round";
  octx.lineWidth = unit * 1.2;
  octx.strokeStyle = "rgba(255,255,255,.75)";
  for (const [a, b] of CONNECTIONS) {
    const A = lm[a], B = lm[b];
    if (!visible(A) || !visible(B)) continue;
    octx.beginPath();
    octx.moveTo(...px(A));
    octx.lineTo(...px(B));
    octx.stroke();
  }
  // Landmarks
  octx.fillStyle = COLORS.teal;
  for (let i = 0; i < lm.length; i++) {
    if (!visible(lm[i])) continue;
    const [x, y] = px(lm[i]);
    octx.beginPath();
    octx.arc(x, y, unit * 1.6, 0, Math.PI * 2);
    octx.fill();
  }

  // Highlight the measurement chain (ear - shoulder - hip) and the vertical reference.
  if (m.valid) {
    const side = m.side;
    const ear = lm[IDX.ear[side]], sh = lm[IDX.shoulder[side]], hip = lm[IDX.hip[side]];
    const over = out.above;
    // vertical reference through the shoulder (dashed)
    octx.setLineDash([unit * 2, unit * 2]);
    octx.lineWidth = unit;
    octx.strokeStyle = "rgba(255,255,255,.8)";
    octx.beginPath();
    octx.moveTo(sh.x * W, Math.max(0, sh.y * H - H * 0.3));
    octx.lineTo(sh.x * W, Math.min(H, sh.y * H + H * 0.3));
    octx.stroke();
    octx.setLineDash([]);
    // neck vector
    octx.lineWidth = unit * 2.4;
    octx.strokeStyle = over ? COLORS.red : COLORS.teal;
    octx.beginPath();
    octx.moveTo(...px(sh));
    octx.lineTo(...px(ear));
    octx.stroke();
    // trunk vector
    octx.strokeStyle = COLORS.blue;
    octx.beginPath();
    octx.moveTo(...px(hip));
    octx.lineTo(...px(sh));
    octx.stroke();
    // key points
    for (const p of [ear, sh, hip]) {
      const [x, y] = px(p);
      octx.fillStyle = COLORS.white;
      octx.beginPath();
      octx.arc(x, y, unit * 2.6, 0, Math.PI * 2);
      octx.fill();
      octx.strokeStyle = over ? COLORS.red : COLORS.teal;
      octx.lineWidth = unit * 0.8;
      octx.stroke();
    }
  }
}

/* ------------------------------------------------------------------ */
/* Live UI                                                             */
/* ------------------------------------------------------------------ */
function setControls() {
  const running = state === "running";
  $("liveStartBtn").disabled = running || state === "starting" || state === "switching";
  $("stopBtn").disabled = !running;
  $("resetBtn").disabled = !running;
  $("switchCamBtn").disabled = state === "starting" || state === "switching";
  $("activePill").classList.toggle("on", running);
  if (!running) $("activeText").textContent = state === "starting" ? "Starting…" : "Not running";
  else $("activeText").textContent = "AI processing active";
}

function resetLiveUI() {
  $("neckValue").textContent = "—";
  $("trunkValue").textContent = "—";
  $("timerText").textContent = "0.0 s";
  $("timerBar").style.width = "0%";
  $("liveElapsed").textContent = "00:00";
  $("liveAlerts").textContent = "0";
  $("liveAbove").textContent = "0.0 s";
  $("alertLog").innerHTML = '<li class="empty">No alerts yet.</li>';
  $("alertBanner").hidden = true;
  $("repositionBanner").hidden = true;
  $("stage").classList.remove("alert");
  setStatus("idle", "—");
  refreshTimerHint();
  setControls();
  octx.clearRect(0, 0, overlay.width, overlay.height);
}

function setStatus(kind, text) {
  $("statusCard").dataset.state = kind;
  $("statusValue").textContent = text;
}

function refreshTimerHint() {
  $("timerHint").textContent =
    `Alert after ${settings.duration} s continuously above ${settings.threshold}° (configurable prototype threshold).`;
}

function updateReadouts(out, m) {
  $("neckValue").textContent = out.valid ? fmtDeg(out.neck) : "—";
  $("trunkValue").textContent = out.valid ? fmtDeg(out.trunk) : "—";
  $("neckMetric").classList.toggle("over", out.valid && out.above);

  $("timerText").textContent = fmtSec(out.consecutiveSec);
  $("timerBar").style.width = `${Math.min(100, (out.consecutiveSec / settings.duration) * 100)}%`;
  document.querySelector(".timer-card").classList.toggle("over", out.valid && out.above);

  $("liveElapsed").textContent = fmtClock(out.elapsedSec);
  $("liveAlerts").textContent = String(out.alertCount);
  $("liveAbove").textContent = fmtSec(out.totalAboveSec);

  if (!out.valid) {
    setStatus("none", "Unable to Detect");
    $("repositionBanner").textContent = `Please reposition: ${m.reason || "no reliable measurement"}.`;
    $("repositionBanner").hidden = false;
  } else {
    $("repositionBanner").hidden = true;
    if (out.alertActive) setStatus("alert", "Warning");
    else if (out.above) setStatus("warn", "Warning");
    else setStatus("good", "Good");
  }
  $("alertBanner").hidden = !out.alertActive;
  $("stage").classList.toggle("alert", out.alertActive);
}

function addAlertToLog(a) {
  const log = $("alertLog");
  const empty = log.querySelector(".empty");
  if (empty) empty.remove();
  const li = document.createElement("li");
  const clock = new Date(a.wallMs).toLocaleTimeString();
  li.textContent = `#${a.n} · ${fmtClock(a.elapsedSec)} elapsed · ${a.angle.toFixed(1)}° · ${clock}`;
  log.prepend(li);
}

/* ------------------------------------------------------------------ */
/* Summary, chart, CSV                                                 */
/* ------------------------------------------------------------------ */
function renderSummary(s, note) {
  $("summaryNotice").hidden = !note;
  $("summaryNotice").textContent = note || "";
  $("sumDuration").textContent = fmtClock(s.durationSec);
  $("sumAlerts").textContent = String(s.alertCount);
  $("sumAbove").textContent = fmtSec(s.timeAboveSec);
  $("sumAvgNeck").textContent = fmtDeg(s.avgNeck);
  $("sumAvgTrunk").textContent = fmtDeg(s.avgTrunk);
  $("sumMaxNeck").textContent = fmtDeg(s.maxNeck);
  $("sumValid").textContent = `${s.validPct.toFixed(0)}%`;
  $("sumSettings").textContent =
    `Settings at end of session: alert when neck inclination > ${s.threshold}° for ${s.requiredDuration} s ` +
    `(configurable prototype threshold, requires ergonomic validation).`;

  const body = $("sumAlertRows");
  body.innerHTML = "";
  if (!s.alerts.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "empty";
    td.textContent = "No alerts were triggered in this session.";
    tr.appendChild(td);
    body.appendChild(tr);
  } else {
    for (const a of s.alerts) {
      const tr = document.createElement("tr");
      [a.n, fmtClock(a.elapsedSec), new Date(a.wallMs).toLocaleTimeString(), fmtDeg(a.angle)].forEach((v) => {
        const td = document.createElement("td");
        td.textContent = String(v);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
  }
}

function drawChart(s) {
  const c = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const cw = c.clientWidth;
  const ch = c.clientHeight;
  if (!cw || !ch) return;
  c.width = Math.round(cw * dpr);
  c.height = Math.round(ch * dpr);
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  ctx.font = "12px -apple-system, Helvetica, Arial, sans-serif";

  const pad = { l: 44, r: 12, t: 12, b: 26 };
  const w = cw - pad.l - pad.r;
  const h = ch - pad.t - pad.b;
  const vals = [];
  for (const p of s.samples) if (p.valid) vals.push(p.neck, p.trunk);

  if (!vals.length) {
    ctx.fillStyle = "#475569";
    ctx.textAlign = "center";
    ctx.fillText("No valid measurements were recorded in this session.", cw / 2, ch / 2);
    return;
  }

  const yMin = Math.floor(Math.min(-10, ...vals) / 10) * 10;
  const yMax = Math.ceil(Math.max(60, s.threshold + 10, ...vals) / 10) * 10;
  const T = Math.max(s.durationSec, 1);
  const X = (t) => pad.l + (t / T) * w;
  const Y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * h;

  // grid + y labels
  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#475569";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  const step = (yMax - yMin) > 80 ? 20 : 10;
  for (let v = yMin; v <= yMax; v += step) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(pad.l + w, Y(v)); ctx.stroke();
    ctx.fillText(`${v}°`, pad.l - 6, Y(v) + 4);
  }
  // x labels
  ctx.textAlign = "center";
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const t = (T * i) / ticks;
    ctx.fillText(fmtClock(t), X(t), ch - 8);
  }
  // threshold
  ctx.strokeStyle = "#b91c1c";
  ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(s.threshold)); ctx.lineTo(pad.l + w, Y(s.threshold)); ctx.stroke();
  ctx.setLineDash([]);

  const line = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let pen = false;
    for (const p of s.samples) {
      if (!p.valid) { pen = false; continue; }
      if (!pen) { ctx.moveTo(X(p.t), Y(p[key])); pen = true; } else ctx.lineTo(X(p.t), Y(p[key]));
    }
    ctx.stroke();
  };
  line("trunk", "#0369a1");
  line("neck", "#0f766e");

  // alert markers
  ctx.fillStyle = "#b91c1c";
  for (const a of s.alerts) {
    ctx.beginPath();
    ctx.arc(X(a.elapsedSec), Y(a.angle), 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function buildCsv(s) {
  const rows = [];
  const add = (...cells) => rows.push(cells.map(csvCell).join(","));
  const num = (v, d = 2) => (v == null ? "" : v.toFixed(d));
  add("PostureGuard AI session summary (educational prototype - not a medical device)");
  add("Metric", "Value");
  add("Session start", new Date(s.startedAt).toISOString());
  add("Duration (s)", num(s.durationSec, 1));
  add("Posture alerts", s.alertCount);
  add("Time above threshold (s)", num(s.timeAboveSec, 1));
  add("Average neck inclination (deg, valid frames only)", num(s.avgNeck));
  add("Average trunk flexion (deg, valid frames only)", num(s.avgTrunk));
  add("Maximum neck inclination (deg)", num(s.maxNeck));
  add("Frames with reliable detection (%)", num(s.validPct, 1));
  add("Threshold setting (deg, prototype value)", s.threshold);
  add("Required duration setting (s)", s.requiredDuration);
  add();
  add("Alerts");
  add("Alert #", "Elapsed (s)", "Clock time", "Neck angle (deg)");
  for (const a of s.alerts) add(a.n, num(a.elapsedSec, 1), new Date(a.wallMs).toISOString(), num(a.angle));
  add();
  add("Measurements over time (numbers only; no images or video)");
  add("Time (s)", "Neck inclination (deg)", "Trunk flexion (deg)", "Reliable");
  for (const p of s.samples) add(num(p.t, 1), p.valid ? num(p.neck) : "", p.valid ? num(p.trunk) : "", p.valid ? "yes" : "no");
  return rows.join("\r\n");
}

function exportCsv() {
  if (!lastSummary) return;
  const blob = new Blob([buildCsv(lastSummary)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const d = new Date(lastSummary.startedAt);
  const p = (n) => String(n).padStart(2, "0");
  const a = document.createElement("a");
  a.href = url;
  a.download = `postureguard-session-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/* ------------------------------------------------------------------ */
/* Settings panel                                                      */
/* ------------------------------------------------------------------ */
function openSettings() { $("settingsModal").hidden = false; }
function closeSettings() { $("settingsModal").hidden = true; }

function syncSettingsLabels() {
  $("thrOut").textContent = `${settings.threshold}°`;
  $("durOut").textContent = `${settings.duration} s`;
  refreshTimerHint();
}

$("thrInput").addEventListener("input", (e) => { settings.threshold = Number(e.target.value); syncSettingsLabels(); });
$("durInput").addEventListener("input", (e) => { settings.duration = Number(e.target.value); syncSettingsLabels(); });
$("soundInput").addEventListener("change", (e) => {
  settings.sound = e.target.checked;
  if (settings.sound) { unlockAudio(); beep(); } // confirmation beep (user gesture unlocks audio on iOS)
});
document.querySelectorAll('input[name="camera"]').forEach((r) =>
  r.addEventListener("change", async (e) => {
    if (!e.target.checked || e.target.value === settings.facing) return;
    const wanted = e.target.value;
    setFacing(wanted === "user" ? "environment" : "user"); // so switchCamera() toggles to `wanted`
    closeSettings();
    await switchCamera();
  })
);
$("settingsCloseBtn").addEventListener("click", closeSettings);
$("settingsModal").addEventListener("click", (e) => { if (e.target === $("settingsModal")) closeSettings(); });

/* ------------------------------------------------------------------ */
/* Wiring                                                              */
/* ------------------------------------------------------------------ */
$("startBtn").addEventListener("click", startMonitoring);
$("liveStartBtn").addEventListener("click", () => { if (state === "idle") startMonitoring(); });
$("stopBtn").addEventListener("click", () => stopMonitoring());
$("resetBtn").addEventListener("click", resetSession);
$("switchCamBtn").addEventListener("click", switchCamera);
$("cancelStartBtn").addEventListener("click", cancelStart);
$("homeSettingsBtn").addEventListener("click", openSettings);
$("liveSettingsBtn").addEventListener("click", openSettings);
$("newSessionBtn").addEventListener("click", () => { clearHomeError(); show("home"); });
$("exportBtn").addEventListener("click", exportCsv);
$("retryModelBtn").addEventListener("click", async () => {
  clearHomeError();
  try { await loadModel(); } catch (e) {
    const info = describeModelError(e);
    showHomeError(info.title, info.body, { retryModel: true });
  }
});

window.addEventListener("resize", () => {
  if (!$("summary").hidden && lastSummary) drawChart(lastSummary);
});
window.addEventListener("orientationchange", () => {
  setTimeout(() => {
    updateStageAspect();
    if (!$("summary").hidden && lastSummary) drawChart(lastSummary);
  }, 300);
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state === "running" && video.paused) {
    video.play().catch(() => {});
  }
});

// Release the camera and model when the page is closed or navigated away.
window.addEventListener("pagehide", () => {
  stopLoop();
  stopCamera();
  disposeModel();
  state = "idle";
});
// Safari can restore a page from its back/forward cache: reload the model if so.
window.addEventListener("pageshow", (e) => {
  if (e.persisted) {
    state = "idle";
    show("home");
    loadModel().catch(() => {});
  }
});

/* ------------------------------------------------------------------ */
/* Init                                                                */
/* ------------------------------------------------------------------ */
syncSettingsLabels();
resetLiveUI();
$("stageLoading").hidden = true;
{
  const unsupported = supportError();
  if (unsupported) {
    showHomeError(unsupported.title, unsupported.body);
    setModelPill("error", "Unsupported browser or insecure page");
  } else {
    // Pre-load the model while the user reads the instructions (camera is NOT touched here).
    loadModel().catch((e) => {
      const info = describeModelError(e);
      showHomeError(info.title, info.body, { retryModel: true });
    });
  }
}
