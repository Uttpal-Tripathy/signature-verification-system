/**
 * SIGNUM app shell — wires the signature pads to the /api/verify endpoint,
 * drives the pipeline HUD, and renders the forensic result panel.
 */
(() => {
  // Always same-origin: api/app.py mounts this static site and the API on one FastAPI app.
  const API_BASE = "";
  // Populated by js/config.js (generated server-side from SIGVERIFY_API_KEY at startup,
  // see api/app.py's _write_frontend_config) -- empty string when auth is disabled, which
  // the backend also treats as "no key required," so this header is always safe to send.
  const apiHeaders = () => ({ "X-API-Key": window.SIGNUM_API_KEY || "" });
  const HUD_STAGES = [
    "REGION LOCALIZATION",
    "PREPROCESSING (denoise / deskew / binarize)",
    "STATIC BRANCH — SIAMESE CNN EMBEDDING",
    "DYNAMIC BRANCH — STROKE ENCODER EMBEDDING",
    "CROSS-ATTENTION FUSION",
    "ANOMALY SCORING",
    "DECISION CALIBRATION",
    "EXPLAINABILITY (Grad-CAM / SHAP)",
  ];

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- state
  const state = {
    refSource: null,       // { kind: 'file'|'pad', blob, strokeData: object|null }
    verifying: false,
  };

  // ---------------------------------------------------------------- pads
  const refPad = new SignaturePad($("refPadCanvas"), { strokeColor: "#ff2d95" });
  const mainPad = new SignaturePad($("mainPadCanvas"), {
    strokeColor: "#00fff2",
    onChange: (tel) => {
      updateTelemetry(tel);
      pollLiveTrend();
      scheduleAutoVerify();
    },
  });

  function updateTelemetry(tel) {
    $("telPoints").textContent = tel.points;
    $("telDuration").textContent = `${tel.durationSec.toFixed(2)}s`;
    $("telPressure").textContent = tel.avgPressure === null ? "—" : tel.avgPressure.toFixed(2);
    $("telVelocity").textContent = tel.avgVelocity === null ? "—" : `${tel.avgVelocity.toFixed(0)} px/s`;
    $("telInputType").textContent = tel.inputType.toUpperCase();
    refreshVerifyEnabled();
  }

  function refreshVerifyEnabled() {
    const hasRef = state.refSource !== null;
    const hasQuery = !mainPad.isEmpty();
    $("verifyBtn").disabled = !(hasRef && hasQuery && !state.verifying);
  }

  // ---------------------------------------------------------------- reference: file upload
  const dropzone = $("refDropzone");
  const fileInput = $("refFileInput");
  const previewImg = $("refPreviewImg");
  const dropInner = $("refDropInner");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setReferenceFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setReferenceFile(fileInput.files[0]);
  });

  function setReferenceFile(file) {
    state.refSource = { kind: "file", blob: file, strokeData: null };
    previewImg.src = URL.createObjectURL(file);
    previewImg.hidden = false;
    dropInner.hidden = true;
    refreshVerifyEnabled();
  }

  // ---------------------------------------------------------------- reference: mini pad
  $("refPadClear").addEventListener("click", () => refPad.clear());
  $("refPadUse").addEventListener("click", async () => {
    if (refPad.isEmpty()) return;
    const blob = await refPad.toBlob();
    state.refSource = { kind: "pad", blob, strokeData: refPad.getStrokeData() };
    previewImg.hidden = true;
    dropInner.hidden = false;
    dropInner.querySelector("p").innerHTML = "REFERENCE CAPTURED FROM PAD<br /><span>click to override with a file</span>";
    refreshVerifyEnabled();
  });

  // ---------------------------------------------------------------- reference: live camera capture
  const cameraVideo = $("refCameraVideo");
  const cameraCanvas = $("refCameraCanvas");
  const cameraPlaceholder = $("cameraPlaceholder");
  let cameraStream = null;

  $("cameraStart").addEventListener("click", async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("Camera capture isn't supported in this browser.");
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      cameraVideo.srcObject = cameraStream;
      cameraVideo.hidden = false;
      cameraPlaceholder.hidden = true;
      $("cameraStart").hidden = true;
      $("cameraCapture").hidden = false;
      $("cameraStop").hidden = false;
    } catch (err) {
      showError(`Camera unavailable: ${err.message || err}`);
    }
  });

  $("cameraCapture").addEventListener("click", () => {
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0);
    cameraCanvas.toBlob((blob) => {
      if (!blob) return;
      state.refSource = { kind: "camera", blob, strokeData: null };
      previewImg.src = URL.createObjectURL(blob);
      previewImg.hidden = false;
      dropInner.hidden = true;
      refreshVerifyEnabled();
      stopCamera();
    }, "image/png");
  });

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }
    cameraVideo.hidden = true;
    cameraPlaceholder.hidden = false;
    $("cameraStart").hidden = false;
    $("cameraCapture").hidden = true;
    $("cameraStop").hidden = true;
  }
  $("cameraStop").addEventListener("click", stopCamera);

  // ---------------------------------------------------------------- main pad
  $("mainPadClear").addEventListener("click", () => {
    mainPad.clear();
    updateTelemetry(mainPad.getTelemetry());
    resetLiveTrend();
    if (autoVerifyTimer) clearTimeout(autoVerifyTimer);
  });

  // ---------------------------------------------------------------- live match trend (while signing)
  // Cheap, dynamic-branch-only similarity polled while the user is actively drawing —
  // only meaningful when the reference sample itself has stroke data (i.e. it was
  // captured via the mini-pad, not uploaded/photographed as a flat image), since an
  // image alone has nothing for the dynamic branch to compare against.
  let liveTrendInFlight = false;
  let liveTrendLastCallAt = 0;
  const LIVE_TREND_MIN_INTERVAL_MS = 600;
  const LIVE_TREND_MIN_POINTS = 10;

  function resetLiveTrend() {
    const item = $("telLiveTrendItem");
    item.classList.remove("trend-up", "trend-down");
    delete item.dataset.prevScore;
    $("telLiveTrend").textContent = "—";
  }

  async function pollLiveTrend() {
    const item = $("telLiveTrendItem");
    if (!state.refSource?.strokeData) {
      item.hidden = true;
      return;
    }
    item.hidden = false;
    if (mainPad.points.length < LIVE_TREND_MIN_POINTS) return;
    const now = performance.now();
    if (liveTrendInFlight || now - liveTrendLastCallAt < LIVE_TREND_MIN_INTERVAL_MS) return;
    liveTrendLastCallAt = now;
    liveTrendInFlight = true;
    try {
      const form = new FormData();
      form.append("reference_stroke", new Blob([JSON.stringify(state.refSource.strokeData)], { type: "application/json" }), "reference_stroke.json");
      form.append("query_stroke", new Blob([JSON.stringify(mainPad.getStrokeData())], { type: "application/json" }), "query_stroke.json");
      const resp = await fetch(`${API_BASE}/api/verify/live`, { method: "POST", body: form, headers: apiHeaders() });
      if (!resp.ok) return;
      const data = await resp.json();
      const prev = Number(item.dataset.prevScore || "0");
      $("telLiveTrend").textContent = data.dynamic_similarity.toFixed(3);
      item.classList.remove("trend-up", "trend-down");
      if (data.dynamic_similarity > prev + 0.01) item.classList.add("trend-up");
      else if (data.dynamic_similarity < prev - 0.01) item.classList.add("trend-down");
      item.dataset.prevScore = String(data.dynamic_similarity);
    } catch {
      // transient network hiccup on a background poll -- not worth surfacing as an error
    } finally {
      liveTrendInFlight = false;
    }
  }

  // ---------------------------------------------------------------- auto-verify on signature pause
  // Detects "signature complete" the same way a human would notice it — the pen has
  // stopped moving for a beat — and runs the full verification automatically, instead
  // of requiring an explicit button click. Manual RUN VERIFICATION still works too.
  let autoVerifyTimer = null;
  const AUTO_VERIFY_DEBOUNCE_MS = 900;
  const AUTO_VERIFY_MIN_POINTS = 20;

  function scheduleAutoVerify() {
    if (autoVerifyTimer) clearTimeout(autoVerifyTimer);
    if (!$("toggleAutoVerify").checked) return;
    if (!state.refSource || mainPad.isEmpty() || state.verifying) return;
    autoVerifyTimer = setTimeout(() => {
      if (!state.verifying && state.refSource && mainPad.points.length >= AUTO_VERIFY_MIN_POINTS) {
        runVerification();
      }
    }, AUTO_VERIFY_DEBOUNCE_MS);
  }

  // ---------------------------------------------------------------- HUD
  let hudTimer = null;

  function startHud() {
    const hud = $("hudLog");
    hud.innerHTML = "";
    $("resultBlock").hidden = true;
    $("errorBlock").hidden = true;

    let i = 0;
    const lines = [];
    const addLine = (text, cls) => {
      const el = document.createElement("div");
      el.className = `hud__line ${cls}`;
      el.innerHTML = `<span class="hud__bullet">&#9679;</span><span class="hud__text">${text}</span>`;
      hud.appendChild(el);
      return el;
    };

    hudTimer = setInterval(() => {
      if (lines.length) lines[lines.length - 1].classList.replace("active", "ok");
      if (i >= HUD_STAGES.length) { clearInterval(hudTimer); return; }
      lines.push(addLine(HUD_STAGES[i], "active"));
      i++;
    }, 420);
  }

  function finishHud(ok) {
    if (hudTimer) clearInterval(hudTimer);
    const hud = $("hudLog");
    const active = hud.querySelector(".hud__line.active");
    if (active) active.classList.replace("active", ok ? "ok" : "err");
    const el = document.createElement("div");
    el.className = `hud__line ${ok ? "ok" : "err"}`;
    el.innerHTML = `<span class="hud__bullet">&#9679;</span><span class="hud__text">${ok ? "VERIFICATION REPORT READY" : "PIPELINE ERROR"}</span>`;
    hud.appendChild(el);
  }

  // ---------------------------------------------------------------- verify
  $("verifyBtn").addEventListener("click", runVerification);

  async function runVerification() {
    if (state.verifying) return;
    state.verifying = true;
    refreshVerifyEnabled();
    startHud();

    try {
      const queryBlob = await mainPad.toBlob();
      const form = new FormData();
      form.append("reference_image", state.refSource.blob, "reference.png");
      form.append("query_image", queryBlob, "query.png");

      if (state.refSource.strokeData) {
        form.append("reference_stroke", new Blob([JSON.stringify(state.refSource.strokeData)], { type: "application/json" }), "reference_stroke.json");
      }
      form.append("query_stroke", new Blob([JSON.stringify(mainPad.getStrokeData())], { type: "application/json" }), "query_stroke.json");

      const writerId = $("writerId").value.trim();
      if (writerId) form.append("user_id", writerId);
      form.append("localize", $("toggleLocalize").checked ? "true" : "false");
      form.append("estimate_confidence", $("toggleConfidence").checked ? "true" : "false");

      const resp = await fetch(`${API_BASE}/api/verify`, { method: "POST", body: form, headers: apiHeaders() });
      if (!resp.ok) {
        const detail = await safeJson(resp);
        throw new Error(detail?.detail || `HTTP ${resp.status}`);
      }
      const result = await resp.json();
      finishHud(true);
      renderResult(result);
    } catch (err) {
      finishHud(false);
      showError(err.message || String(err));
    } finally {
      state.verifying = false;
      refreshVerifyEnabled();
    }
  }

  async function safeJson(resp) {
    try { return await resp.json(); } catch { return null; }
  }

  function showError(message) {
    const el = $("errorBlock");
    el.hidden = false;
    el.textContent = `⚠ ${message}`;
  }

  // ---------------------------------------------------------------- results
  function renderResult(r) {
    $("resultBlock").hidden = false;

    const badge = $("decisionBadge");
    badge.className = "result__badge " + r.decision.toLowerCase();
    $("decisionLabel").textContent = r.decision.toUpperCase();

    const score = clamp01(r.combined_score);
    $("gaugeValue").textContent = score.toFixed(3);
    const circumference = 173; // matches the SVG arc's path length approximation
    const fill = $("gaugeFill");
    fill.style.strokeDashoffset = String(circumference * (1 - score));
    fill.style.stroke = score >= 0.8 ? "var(--green)" : score >= 0.55 ? "var(--amber)" : "var(--red)";

    const staticPct = Math.round((r.modality_weights?.static_weight ?? 0) * 100);
    const dynamicPct = Math.round((r.modality_weights?.dynamic_weight ?? 0) * 100);
    $("mwStaticFill").style.width = `${staticPct}%`;
    $("mwDynamicFill").style.width = `${dynamicPct}%`;
    $("mwStaticPct").textContent = `${staticPct}%`;
    $("mwDynamicPct").textContent = `${dynamicPct}%`;

    $("mFused").textContent = fmt(r.fused_similarity);
    $("mStatic").textContent = fmt(r.static_similarity);
    $("mDynamic").textContent = fmt(r.dynamic_similarity);
    $("mCalibrated").textContent = fmt(r.calibrated_score);
    $("mAnomaly").textContent = fmt(r.anomaly_score);
    $("mCI").textContent = r.confidence_interval
      ? `[${r.confidence_interval[0].toFixed(3)}, ${r.confidence_interval[1].toFixed(3)}]`
      : "—";

    const heatmapBlock = $("heatmapBlock");
    if (r.static_heatmap_png_base64) {
      $("heatmapImg").src = `data:image/png;base64,${r.static_heatmap_png_base64}`;
      heatmapBlock.hidden = false;
    } else {
      heatmapBlock.hidden = true;
    }

    const shapBlock = $("shapBlock");
    const shapBars = $("shapBars");
    if (r.shap_modality_split) {
      shapBars.innerHTML = "";
      for (const [key, val] of Object.entries(r.shap_modality_split)) {
        const row = document.createElement("div");
        row.className = "shap-row";
        row.innerHTML = `<span class="shap-row__label">${key.replace(/_/g, " ").replace("contribution pct", "").trim()}</span>
          <span class="shap-row__bar"><span style="width:${val}%"></span></span>
          <span class="shap-row__pct">${val.toFixed(1)}%</span>`;
        shapBars.appendChild(row);
      }
      shapBlock.hidden = false;
    } else {
      shapBlock.hidden = true;
    }
  }

  function fmt(v) { return v === null || v === undefined ? "—" : Number(v).toFixed(4); }
  function clamp01(v) { return Math.max(0, Math.min(1, v)); }

  // ---------------------------------------------------------------- health + clock
  async function checkHealth() {
    const dot = $("apiStatusDot");
    const text = $("apiStatusText");
    try {
      const resp = await fetch(`${API_BASE}/api/health`, { headers: apiHeaders() });
      const data = await resp.json();
      dot.className = "status-dot online";
      text.textContent = `ONLINE · ${data.device.toUpperCase()}`;
    } catch {
      dot.className = "status-dot offline";
      text.textContent = "BACKEND UNREACHABLE";
    }
  }

  function tickClock() {
    $("clock").textContent = new Date().toTimeString().slice(0, 8);
  }

  // ---------------------------------------------------------------- init
  checkHealth();
  setInterval(checkHealth, 15000);
  tickClock();
  setInterval(tickClock, 1000);
  refreshVerifyEnabled();
})();
