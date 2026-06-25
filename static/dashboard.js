/* NeuroGuard v2.0 Web — Dashboard JS with WebRTC frame capture */
"use strict";

// ── Session ID (unique per browser tab) ──────────────────────────────────────
const SESSION_ID = "ws_" + Math.random().toString(36).slice(2, 10);
let _active      = false;
let _fpsCount    = 0;
let _fpsVal      = 0;
let _fpsInterval = null;
let _captureInterval = null;
let _stream      = null;

// ── SocketIO ──────────────────────────────────────────────────────────────────
const socket = io();
socket.on("connect",    () => connDot(true));
socket.on("disconnect", () => connDot(false));

function connDot(live) {
  const el = document.getElementById("conn-dot");
  el.className = live ? "live" : "";
}

// ── Camera controls ───────────────────────────────────────────────────────────
const video   = document.getElementById("webcam");
const capCvs  = document.getElementById("capture-canvas");
const preview = document.getElementById("cam-preview");
const capCtx  = capCvs.getContext("2d");
const preCtx  = preview.getContext("2d");

async function startCamera() {
  try {
    _stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = _stream;
    await video.play();

    // Size canvases
    capCvs.width  = preview.width  = 640;
    capCvs.height = preview.height = 480;
    preview.style.width  = "100%";
    preview.style.borderRadius = "8px";

    _active = true;
    document.getElementById("btn-start").disabled = true;
    document.getElementById("btn-stop").disabled  = false;
    document.getElementById("cam-status").textContent = "Camera active — monitoring...";
    document.getElementById("cam-status").style.color = "var(--accent)";
    document.getElementById("status-badge").textContent = "ACTIVE";
    document.getElementById("status-badge").className = "badge normal";

    // Capture frames at ~10 fps
    _captureInterval = setInterval(captureAndSend, 100);
    _fpsInterval     = setInterval(() => { _fpsVal = _fpsCount; _fpsCount = 0; }, 1000);

  } catch (err) {
    document.getElementById("cam-status").textContent = "Camera error: " + err.message;
    document.getElementById("cam-status").style.color = "var(--critical)";
  }
}

function stopCamera() {
  _active = false;
  clearInterval(_captureInterval);
  clearInterval(_fpsInterval);
  if (_stream) _stream.getTracks().forEach(t => t.stop());

  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").disabled  = true;
  document.getElementById("cam-status").textContent = "Camera stopped";
  document.getElementById("cam-status").style.color = "var(--muted)";
  document.getElementById("status-badge").textContent = "STANDBY";
  document.getElementById("status-badge").className = "badge normal";

  // End session on server
  fetch("/api/end_session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION_ID }),
  });
}

let _sending = false;

async function captureAndSend() {
  if (!_active || _sending || video.readyState < 2) return;
  _sending = true;

  try {
    // Draw to hidden canvas, then to visible preview (mirrored)
    capCtx.drawImage(video, 0, 0, 640, 480);

    // Mirror the preview
    preCtx.save();
    preCtx.translate(640, 0);
    preCtx.scale(-1, 1);
    preCtx.drawImage(video, 0, 0, 640, 480);
    preCtx.restore();

    const frameB64 = capCvs.toDataURL("image/jpeg", 0.7);

    const res = await fetch("/api/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame:      frameB64,
        session_id: SESSION_ID,
        driver_id:  "web_user",
      }),
    });

    if (res.ok) {
      const m = await res.json();
      updateDashboard(m);
      _fpsCount++;
    }
  } catch (e) {
    // network error — silent, will retry
  } finally {
    _sending = false;
  }
}

// ── Chart defaults ────────────────────────────────────────────────────────────
Chart.defaults.color = "#4a6080";
Chart.defaults.font.family = "'Inter', sans-serif";

const CHART_OPTS = {
  animation: false,
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { display: false },
    y: { grid: { color: "rgba(0,255,153,0.04)" }, ticks: { color: "#4a6080", font: { size: 9 } } },
  },
};
const MAX_PTS = 80;

const fatigueData  = { labels: [], datasets: [{ label: "Fatigue %", data: [], borderColor: "#00ff99", backgroundColor: "rgba(0,255,153,0.06)", borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 }] };
const attnData     = { labels: [], datasets: [{ label: "Attention", data: [], borderColor: "#00c8ff", backgroundColor: "rgba(0,200,255,0.05)", borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 }] };
const blinkData    = { labels: [], datasets: [{ label: "Blinks/min", data: [], borderColor: "#7b5fff", backgroundColor: "rgba(123,95,255,0.05)", borderWidth: 2, tension: 0.3, fill: true, pointRadius: 0 }] };

const fatigueChart  = new Chart(document.getElementById("fatigueChart"),  { type: "line", data: fatigueData, options: { ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0, max: 100 } } } });
const attentionChart= new Chart(document.getElementById("attentionChart"), { type: "line", data: attnData,    options: { ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0, max: 1  } } } });
const blinkChart    = new Chart(document.getElementById("blinkChart"),     { type: "line", data: blinkData,   options: { ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0       } } } });

// ── Gaze Radar ────────────────────────────────────────────────────────────────
const gazeCanvas = document.getElementById("gazeCanvas");
const gazeCtx    = gazeCanvas.getContext("2d");

function drawGaze(gx, gy) {
  const W = gazeCanvas.width, H = gazeCanvas.height, cx = W/2, cy = H/2, r = W/2 - 4;
  gazeCtx.clearRect(0, 0, W, H);
  for (let i = 3; i >= 1; i--) {
    gazeCtx.beginPath(); gazeCtx.arc(cx, cy, r * i / 3, 0, Math.PI*2);
    gazeCtx.strokeStyle = "rgba(0,255,153,0.07)"; gazeCtx.lineWidth = 1; gazeCtx.stroke();
  }
  gazeCtx.strokeStyle = "rgba(0,255,153,0.1)"; gazeCtx.lineWidth = 1;
  gazeCtx.beginPath(); gazeCtx.moveTo(cx, 4); gazeCtx.lineTo(cx, H-4); gazeCtx.stroke();
  gazeCtx.beginPath(); gazeCtx.moveTo(4, cy); gazeCtx.lineTo(W-4, cy); gazeCtx.stroke();
  const px = cx + gx * r * 2, py = cy + gy * r * 2;
  const dev = Math.sqrt(gx*gx + gy*gy), c = dev > 0.25 ? "#ff2d4b" : "#00ff99";
  gazeCtx.beginPath(); gazeCtx.arc(px, py, 12, 0, Math.PI*2);
  gazeCtx.fillStyle = dev > 0.25 ? "rgba(255,45,75,0.1)" : "rgba(0,255,153,0.08)"; gazeCtx.fill();
  gazeCtx.beginPath(); gazeCtx.arc(px, py, 5, 0, Math.PI*2);
  gazeCtx.fillStyle = c; gazeCtx.shadowColor = c; gazeCtx.shadowBlur = 10; gazeCtx.fill(); gazeCtx.shadowBlur = 0;
}
drawGaze(0, 0);

// ── SVG Gauge ─────────────────────────────────────────────────────────────────
const CIRC = 2 * Math.PI * 80;
const gaugeFill = document.getElementById("gauge-fill");
const gaugePct  = document.getElementById("gauge-pct");

function updateGauge(pct) {
  const c = Math.min(100, Math.max(0, pct));
  gaugeFill.style.strokeDashoffset = CIRC * (1 - c/100);
  const col = c < 30 ? "#00ff99" : c < 55 ? "#ffc93c" : c < 75 ? "#ff7b2f" : "#ff2d4b";
  gaugeFill.style.stroke = col; gaugePct.style.color = col;
  gaugePct.textContent = c.toFixed(0) + "%";
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const LEVEL_COLORS  = ["#00ff99","#ffc93c","#ff7b2f","#ff2d4b"];
const LEVEL_CLASSES = ["normal","warning","danger","critical"];
const LEVEL_LABELS  = ["NORMAL","WARNING","DANGER","CRITICAL"];

function applyLevel(level) {
  const badge = document.getElementById("status-badge");
  badge.textContent = LEVEL_LABELS[level] || "NORMAL";
  badge.className   = `badge ${LEVEL_CLASSES[level] || "normal"}`;
}

function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setBar(id, pct, col) { const el = document.getElementById(id); if (!el) return; el.style.width = Math.min(100, Math.max(0, pct)) + "%"; if (col) el.style.background = col; }
function pushChart(data, chart, val) {
  const now = new Date().toLocaleTimeString("en-GB", { hour12: false });
  data.labels.push(now); data.datasets[0].data.push(val);
  if (data.labels.length > MAX_PTS) { data.labels.shift(); data.datasets[0].data.shift(); }
  chart.update("none");
}

// ── Session tracking ──────────────────────────────────────────────────────────
let _sessionStart = null, _fatigueSamples = [], _sessionAlerts = 0, _sessionDrowsy = 0, _lastLevel = 0, _lastAlertTs = 0;

function updateSessionStats(m) {
  if (!_sessionStart) _sessionStart = Date.now();
  const sec = Math.round((Date.now() - _sessionStart) / 1000);
  setText("st-duration", sec > 60 ? `${Math.floor(sec/60)}m ${sec%60}s` : `${sec}s`);
  _fatigueSamples.push(+(m.fatigue_score||0));
  if (m.alert_level > 0) _sessionAlerts++;
  if (m.is_drowsy) _sessionDrowsy++;
  const avg = _fatigueSamples.length ? (_fatigueSamples.reduce((a,b)=>a+b,0)/_fatigueSamples.length).toFixed(1) : 0;
  setText("st-avgfat",  avg + "%");
  setText("st-blinks",  m.blink_count || 0);
  setText("st-yawns",   m.yawn_count  || 0);
  setText("st-drowsy",  _sessionDrowsy);
  setText("st-alerts",  _sessionAlerts);
  setText("st-face",    m.face_detected ? "YES" : "NO");
  setText("hdr-blinks", m.blink_count || 0);
  setText("hdr-yawns",  m.yawn_count  || 0);
  setText("fps-val",    _fpsVal + " fps");
  setText("session-time", Math.round(m.session_time||0) + "s");
}

// ── Alert log ─────────────────────────────────────────────────────────────────
function appendAlert(level, m) {
  const now = Date.now();
  if (now - _lastAlertTs < 3000 && level === _lastLevel) return;
  _lastAlertTs = now; _lastLevel = level;
  const ul = document.getElementById("alert-log"), li = document.createElement("li");
  li.className = `lv${level}`;
  li.textContent = `${new Date().toLocaleTimeString()}  [${LEVEL_LABELS[level]}]  FAT=${(m.fatigue_score||0).toFixed(0)}%  EAR=${(m.ear_avg||0).toFixed(3)}`;
  ul.prepend(li);
  while (ul.children.length > 30) ul.removeChild(ul.lastChild);
  // Browser beep for alerts
  playBeep(level);
  // Add to ticker
  const inner = document.getElementById("ticker-inner");
  const span  = document.createElement("span");
  span.className = `ticker-item lv${level}`;
  span.textContent = `${new Date().toLocaleTimeString()} [${LEVEL_LABELS[level]}] FAT=${(m.fatigue_score||0).toFixed(0)}%`;
  inner.prepend(span);
}

// Browser-side audio beep (replaces server sounddevice)
function playBeep(level) {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = level >= 3 ? 1200 : level >= 2 ? 880 : 440;
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    osc.start(); osc.stop(ctx.currentTime + 0.4);
  } catch(_) {}
}

// ── Main metrics handler ──────────────────────────────────────────────────────
function updateDashboard(m) {
  const fatigue = +(m.fatigue_score || 0);
  const level   = +(m.alert_level   || 0);
  const ear     = +(m.ear_avg       || 0);
  const mar     = +(m.mar           || 0);
  const pitch   = +(m.pitch         || 0);
  const blink   = +(m.blink_rate    || 0);
  const attn    = m.attention !== undefined ? +m.attention : null;

  setText("val-ear",   ear.toFixed(3));
  setText("val-mar",   mar.toFixed(3));
  setText("val-pitch", Math.abs(pitch).toFixed(1) + "°");
  setText("val-blink", blink.toFixed(1));
  setText("val-yawn",  m.yawn_count || 0);
  setText("val-attn",  attn !== null ? attn.toFixed(2) : "--");

  setBar("bar-ear",   (ear/0.45)*100);
  setBar("bar-mar",   (mar/1.0)*100, mar>0.7 ? "var(--danger)" : "var(--accent2)");
  setBar("bar-pitch", (Math.abs(pitch)/30)*100, Math.abs(pitch)>20 ? "var(--warn)" : "var(--accent3)");
  setBar("bar-blink", (blink/25)*100);
  setBar("bar-yawn",  Math.min((m.yawn_count||0)*10,100), "var(--warn)");
  if (attn !== null) setBar("bar-attn", attn*100, attn<0.4 ? "var(--critical)" : "var(--accent)");

  updateGauge(fatigue);
  applyLevel(level);

  if (m.gaze_x !== undefined) {
    drawGaze(+m.gaze_x, +m.gaze_y);
    setText("gaze-x", (+m.gaze_x).toFixed(3));
    setText("gaze-y", (+m.gaze_y).toFixed(3));
    const away = Math.sqrt(m.gaze_x**2 + m.gaze_y**2) > 0.25;
    const gs = document.getElementById("gaze-status");
    if (gs) { gs.textContent = away ? "looking away" : "centered"; gs.style.color = away ? "var(--danger)" : "var(--accent)"; }
  }

  if (m.minutes_to_fatigue !== undefined) {
    const mins = +m.minutes_to_fatigue;
    setText("pred-time", mins < 0 ? "Not enough data" : mins === 0 ? "NOW" : mins + " min");
    setText("pred-conf", m.confidence !== undefined ? (+m.confidence*100).toFixed(0)+"%" : "--");
    setText("pred-rec",  m.recommendation || "");
  }

  // Card glow
  document.querySelectorAll(".card").forEach(c => c.classList.remove("alert-glow"));
  if (level >= 3) { ["card-ear","card-pitch"].forEach(id => document.getElementById(id)?.classList.add("alert-glow")); }

  pushChart(fatigueData, fatigueChart, fatigue);
  if (attn !== null) pushChart(attnData, attentionChart, attn);
  pushChart(blinkData, blinkChart, blink);

  if (level > 0) appendAlert(level, m);
  updateSessionStats(m);
}

// ── SocketIO metrics (for multi-tab sync) ─────────────────────────────────────
socket.on("metrics", (m) => {
  // Only update if this tab isn't the one sending (avoid double update)
  if (!_active) updateDashboard(m);
});

// ── Report download ───────────────────────────────────────────────────────────
function downloadReport() {
  const sid = SESSION_ID;
  const link = document.createElement("a");
  link.href = `/api/report/${sid}`;
  link.download = `neuroguard_report_${sid}.png`;
  link.click();
}
