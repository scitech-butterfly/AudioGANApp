/* ── AudioGAN Voice Enhancer – script.js ─────────────────────────────────── */

const API_URL = '/enhance';

// ── DOM refs ──────────────────────────────────────────────────────────────
const recordBtn      = document.getElementById('recordBtn');
const recordBtnText  = document.getElementById('recordBtnText');
const micIcon        = document.getElementById('micIcon');
const stopIcon       = document.getElementById('stopIcon');
const recBadge       = document.getElementById('recBadge');
const recordTimer    = document.getElementById('recordTimer');
const timerDisplay   = document.getElementById('timerDisplay');
const enhanceBtn     = document.getElementById('enhanceBtn');

const inputPlayerWrap  = document.getElementById('inputPlayerWrap');
const inputPlayer      = document.getElementById('inputPlayer');
const outputPlayerWrap = document.getElementById('outputPlayerWrap');
const outputPlayer     = document.getElementById('outputPlayer');

const statusPanel      = document.getElementById('statusPanel');
const statusIconWrap   = document.getElementById('statusIconWrap');
const statusLabel      = document.getElementById('statusLabel');
const statusSub        = document.getElementById('statusSub');
const iconIdle         = document.getElementById('iconIdle');
const iconSpinner      = document.getElementById('iconSpinner');
const iconDone         = document.getElementById('iconDone');
const iconError        = document.getElementById('iconError');

const progressWrap     = document.getElementById('progressWrap');
const progressBar      = document.getElementById('progressBar');
const progressPct      = document.getElementById('progressPct');
const procTimeWrap     = document.getElementById('procTimeWrap');
const procTime         = document.getElementById('procTime');

const metricsSection   = document.getElementById('metricsSection');
const specSection      = document.getElementById('specSection');

const pesqVal = document.getElementById('pesqVal');
const stoiVal = document.getElementById('stoiVal');
const snrVal  = document.getElementById('snrVal');
const pesqBar = document.getElementById('pesqBar');
const stoiBar = document.getElementById('stoiBar');
const snrBar  = document.getElementById('snrBar');

const waveCanvas    = document.getElementById('waveCanvas');
const origSpecCanvas = document.getElementById('origSpecCanvas');
const enhSpecCanvas  = document.getElementById('enhSpecCanvas');

// ── State ─────────────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks   = [];
let recordedBlob  = null;
let isRecording   = false;
let timerInterval = null;
let timerSeconds  = 0;
let audioCtx      = null;
let analyser      = null;
let waveAnimFrame = null;

// ── Waveform visualizer ───────────────────────────────────────────────────
function initVisualizer(stream) {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);
  drawWave();
}

function drawWave() {
  if (!analyser) return;
  const ctx    = waveCanvas.getContext('2d');
  const W      = waveCanvas.clientWidth * window.devicePixelRatio;
  const H      = waveCanvas.clientHeight * window.devicePixelRatio;
  waveCanvas.width  = W;
  waveCanvas.height = H;

  const bufLen = analyser.frequencyBinCount;
  const data   = new Uint8Array(bufLen);
  analyser.getByteTimeDomainData(data);

  ctx.clearRect(0, 0, W, H);

  // gradient line
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0,   'rgba(14,165,233,0.2)');
  grad.addColorStop(0.5, 'rgba(56,189,248,1)');
  grad.addColorStop(1,   'rgba(52,211,153,0.4)');

  ctx.lineWidth   = 2 * window.devicePixelRatio;
  ctx.strokeStyle = grad;
  ctx.beginPath();
  const sliceW = W / bufLen;
  let x = 0;
  for (let i = 0; i < bufLen; i++) {
    const v = data[i] / 128.0;
    const y = (v * H) / 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    x += sliceW;
  }
  ctx.stroke();

  waveAnimFrame = requestAnimationFrame(drawWave);
}

function stopVisualizer() {
  if (waveAnimFrame) { cancelAnimationFrame(waveAnimFrame); waveAnimFrame = null; }
  analyser = null;
  drawIdleWave();
}

function drawIdleWave() {
  const ctx = waveCanvas.getContext('2d');
  const W   = waveCanvas.clientWidth  * window.devicePixelRatio || 420;
  const H   = waveCanvas.clientHeight * window.devicePixelRatio || 80;
  waveCanvas.width  = W;
  waveCanvas.height = H;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(148,163,184,0.15)';
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

// ── Timer ─────────────────────────────────────────────────────────────────
function startTimer() {
  timerSeconds = 0;
  timerDisplay.textContent = '0:00';
  recordTimer.classList.remove('hidden');
  timerInterval = setInterval(() => {
    timerSeconds++;
    const m = Math.floor(timerSeconds / 60);
    const s = String(timerSeconds % 60).padStart(2, '0');
    timerDisplay.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  recordTimer.classList.add('hidden');
}

// ── Recording ─────────────────────────────────────────────────────────────
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    audioChunks = [];

    // Prefer webm/opus (most browsers), fallback to webm
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.onstop = onRecordingStop;
    mediaRecorder.start(100); // collect every 100 ms for lower latency

    isRecording = true;
    setRecordingUI(true);
    initVisualizer(stream);
    startTimer();
  } catch (err) {
    alert(`Microphone access denied or unavailable:\n${err.message}`);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    isRecording = false;
    setRecordingUI(false);
    stopVisualizer();
    stopTimer();
  }
}

function onRecordingStop() {
  const mimeType = mediaRecorder.mimeType || 'audio/webm';
  recordedBlob = new Blob(audioChunks, { type: mimeType });

  const url = URL.createObjectURL(recordedBlob);
  inputPlayer.src = url;
  inputPlayerWrap.classList.remove('hidden');
  enhanceBtn.disabled = false;

  setStatus('idle', 'Ready', 'Click "Enhance Audio" to process');
}

function setRecordingUI(recording) {
  recordBtn.setAttribute('aria-pressed', String(recording));
  recordBtnText.textContent = recording ? 'Stop Recording' : 'Start Recording';
  micIcon.classList.toggle('hidden',  recording);
  stopIcon.classList.toggle('hidden', !recording);
  recBadge.classList.toggle('hidden', !recording);
  recordBtn.classList.toggle('recording', recording);
}

recordBtn.addEventListener('click', () => {
  if (isRecording) stopRecording();
  else startRecording();
});

// ── Status helper ─────────────────────────────────────────────────────────
function setStatus(state, label, sub) {
  statusLabel.textContent = label;
  statusSub.textContent   = sub;

  iconIdle.classList.add('hidden');
  iconSpinner.classList.add('hidden');
  iconDone.classList.add('hidden');
  iconError.classList.add('hidden');
  statusIconWrap.className = 'status-icon-wrap';
  statusPanel.className    = 'status-panel';

  if (state === 'idle') {
    iconIdle.classList.remove('hidden');
  } else if (state === 'processing') {
    iconSpinner.classList.remove('hidden');
    statusIconWrap.classList.add('processing');
    statusPanel.classList.add('processing');
  } else if (state === 'done') {
    iconDone.classList.remove('hidden');
    statusIconWrap.classList.add('done');
    statusPanel.classList.add('done');
  } else if (state === 'error') {
    iconError.classList.remove('hidden');
    statusIconWrap.classList.add('error');
    statusPanel.classList.add('error');
  }
}

// ── Fake progress animation ───────────────────────────────────────────────
let progressInterval = null;

function startProgress() {
  progressWrap.classList.remove('hidden');
  let pct = 0;
  progressBar.style.width = '0%';
  progressPct.textContent  = '0%';
  progressInterval = setInterval(() => {
    // Accelerate quickly then slow toward 90%
    const step = pct < 40 ? 4 : pct < 75 ? 2 : 0.4;
    pct = Math.min(pct + step, 90);
    progressBar.style.width = `${pct}%`;
    progressPct.textContent  = `${Math.round(pct)}%`;
  }, 120);
}

function finishProgress() {
  clearInterval(progressInterval);
  progressBar.style.width = '100%';
  progressPct.textContent  = '100%';
  setTimeout(() => progressWrap.classList.add('hidden'), 700);
}

// ── Enhance ───────────────────────────────────────────────────────────────
enhanceBtn.addEventListener('click', async () => {
  if (!recordedBlob) return;

  enhanceBtn.disabled = true;
  setStatus('processing', 'Processing', 'Running GAN enhancement pipeline…');
  startProgress();
  procTimeWrap.classList.add('hidden');

  const fd = new FormData();
  fd.append('audio', recordedBlob, 'recording.webm');

  const t0 = performance.now();

  try {
    const res = await fetch(API_URL, { method: 'POST', body: fd });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(errData.error || `Server error ${res.status}`);
    }

    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);

    finishProgress();

    // ── Audio playback ──────────────────────────────────────────────────
    if (data.enhanced_audio) {
      outputPlayer.src = `data:audio/wav;base64,${data.enhanced_audio}`;
      outputPlayerWrap.classList.remove('hidden');
    }

    // ── Metrics ─────────────────────────────────────────────────────────
    if (data.metrics) {
      updateMetrics(data.metrics);
      metricsSection.classList.remove('hidden');
    }

    // ── Spectrograms ─────────────────────────────────────────────────────
    if (data.spectrogram) {
      drawSpectrogram(origSpecCanvas, data.spectrogram.original);
      drawSpectrogram(enhSpecCanvas,  data.spectrogram.enhanced);
      specSection.classList.remove('hidden');
    }

    // ── Timing chip ─────────────────────────────────────────────────────
    procTime.textContent = elapsed;
    procTimeWrap.classList.remove('hidden');

    setStatus('done', 'Enhancement Complete', `Processed in ${elapsed} ms`);

  } catch (err) {
    finishProgress();
    setStatus('error', 'Error', err.message || 'Something went wrong');
    alert(`Enhancement failed:\n${err.message}`);
  } finally {
    enhanceBtn.disabled = false;
  }
});

// ── Metrics update ────────────────────────────────────────────────────────
function updateMetrics({ pesq_enhanced, stoi_enhanced, snr_after }) {
  // PESQ 1–4.5 scale
  pesqVal.textContent = pesq_enhanced.toFixed(2);
  pesqBar.style.width = `${Math.min(100, ((pesq_enhanced - 1) / 3.5) * 100).toFixed(1)}%`;

  // STOI 0–1 scale
  stoiVal.textContent = stoi_enhanced.toFixed(3);
  stoiBar.style.width = `${(Math.max(0, Math.min(1, stoi_enhanced)) * 100).toFixed(1)}%`;

  // SNR: treat 0–40 dB as 0–100%
  snrVal.textContent = `${snr_after.toFixed(1)}`;
  snrBar.style.width = `${Math.min(100, Math.max(0, (snr_after / 40) * 100)).toFixed(1)}%`;

  // Animate in metric cards
  document.querySelectorAll('.metric-card').forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(12px)';
    setTimeout(() => {
      card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, i * 100);
  });
}

// ── Spectrogram renderer ──────────────────────────────────────────────────
function drawSpectrogram(canvas, data) {
  if (!data || !data.length) return;

  const rows = data.length;
  const cols = data[0].length;
  canvas.width  = cols;
  canvas.height = rows;

  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);

  // Color map: dark blue → cyan → yellow → white (viridis-inspired)
  function valueToRgb(v) {
    // v in [0,1]
    if (v < 0.25) {
      const t = v / 0.25;
      return [Math.round(t * 0),   Math.round(t * 100), Math.round(80 + t * 130)];
    } else if (v < 0.5) {
      const t = (v - 0.25) / 0.25;
      return [Math.round(t * 0),   Math.round(100 + t * 155), Math.round(210 - t * 80)];
    } else if (v < 0.75) {
      const t = (v - 0.5) / 0.25;
      return [Math.round(t * 255), Math.round(255 - t * 55),  Math.round(130 - t * 100)];
    } else {
      const t = (v - 0.75) / 0.25;
      return [255, Math.round(200 + t * 55), Math.round(30 + t * 225)];
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v   = data[rows - 1 - r][c]; // flip vertically (low freq at bottom)
      const [R, G, B] = valueToRgb(Math.max(0, Math.min(1, v)));
      const idx = (r * cols + c) * 4;
      img.data[idx]     = R;
      img.data[idx + 1] = G;
      img.data[idx + 2] = B;
      img.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
}

// ── Init ──────────────────────────────────────────────────────────────────
(function init() {
  drawIdleWave();
  setStatus('idle', 'Waiting', 'Record audio to begin');

  // Resize waveCanvas on window resize
  window.addEventListener('resize', () => {
    if (!isRecording) drawIdleWave();
  });
})();
