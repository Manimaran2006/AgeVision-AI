"""
Age Prediction Web Application
Flask backend serving the trained model through a browser UI
"""

import os
import sys
import io
import base64
import json
from pathlib import Path

# Ensure inference.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

# Global predictor instance (loaded once at startup)
predictor = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_age_model.pt")
METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final_metrics.json")

def load_predictor():
    global predictor
    if predictor is None:
        from inference import AgePredictor
        predictor = AgePredictor(model_path=MODEL_PATH, metrics_path=METRICS_PATH)
        print(f"[OK] Model loaded: {MODEL_PATH}")

# ─────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgeVision AI — Age Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-deep:       #080c14;
    --bg-card:       #0e1420;
    --bg-glass:      rgba(255,255,255,0.04);
    --border:        rgba(255,255,255,0.08);
    --accent1:       #6c63ff;
    --accent2:       #00d4ff;
    --accent-glow:   rgba(108,99,255,0.35);
    --text-primary:  #f0f4ff;
    --text-muted:    #7b8aab;
    --success:       #00e5a0;
    --warn:          #ffba00;
    --danger:        #ff5c7c;
    --radius:        18px;
    --transition:    0.28s cubic-bezier(.4,0,.2,1);
  }

  html, body {
    min-height: 100vh;
    background: var(--bg-deep);
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text-primary);
    overflow-x: hidden;
  }

  /* ── Animated Background ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(ellipse 80% 60% at 20% 10%, rgba(108,99,255,0.12) 0%, transparent 60%),
      radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0,212,255,0.08) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  /* ── Layout ── */
  .app { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column; }

  header {
    padding: 28px 40px;
    display: flex;
    align-items: center;
    gap: 14px;
    border-bottom: 1px solid var(--border);
    background: rgba(8,12,20,0.8);
    backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 100;
  }

  .logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 24px var(--accent-glow);
    flex-shrink: 0;
  }

  .logo-text { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.4px; }
  .logo-text span { background: linear-gradient(90deg, var(--accent1), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

  .header-badge {
    margin-left: auto;
    background: rgba(0,229,160,0.12);
    border: 1px solid rgba(0,229,160,0.3);
    color: var(--success);
    font-size: 0.72rem; font-weight: 600;
    padding: 4px 12px;
    border-radius: 100px;
    letter-spacing: 0.5px;
  }

  main {
    flex: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 60px 24px 80px;
    width: 100%;
  }

  /* ── Hero ── */
  .hero { text-align: center; margin-bottom: 56px; }
  .hero h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -1px;
    margin-bottom: 16px;
    background: linear-gradient(160deg, #fff 40%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { font-size: 1.05rem; color: var(--text-muted); max-width: 540px; margin: 0 auto; line-height: 1.7; }

  /* ── Model Stats Strip ── */
  .stats-strip {
    display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
    margin-bottom: 52px;
  }
  .stat-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 8px 18px;
    font-size: 0.82rem;
    color: var(--text-muted);
    backdrop-filter: blur(10px);
  }
  .stat-pill strong { color: var(--text-primary); font-weight: 600; }
  .stat-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent1); flex-shrink: 0; }
  .stat-dot.green { background: var(--success); }

  /* ── Two-Column Layout ── */
  .workspace {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 780px) { .workspace { grid-template-columns: 1fr; } }

  /* ── Card ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color var(--transition);
  }
  .card:hover { border-color: rgba(108,99,255,0.3); }

  .card-header {
    padding: 20px 24px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
  }
  .card-icon { font-size: 1.1rem; }
  .card-title { font-size: 0.92rem; font-weight: 600; color: var(--text-primary); letter-spacing: -0.2px; }
  .card-body { padding: 24px; }

  /* ── Upload Zone ── */
  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 40px 24px;
    text-align: center;
    cursor: pointer;
    transition: all var(--transition);
    position: relative;
    background: var(--bg-glass);
    min-height: 220px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent1);
    background: rgba(108,99,255,0.07);
  }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }

  .upload-icon {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(0,212,255,0.15));
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 4px;
  }
  .upload-label { font-size: 0.95rem; font-weight: 500; color: var(--text-primary); }
  .upload-sub { font-size: 0.78rem; color: var(--text-muted); }

  /* ── Preview Image ── */
  #preview-container { display: none; margin-top: 20px; }
  #preview-img {
    width: 100%; border-radius: 12px;
    object-fit: cover; max-height: 300px;
    border: 1px solid var(--border);
    display: block;
  }

  /* ── Button ── */
  .btn-predict {
    width: 100%; margin-top: 18px;
    padding: 15px 24px;
    border: none; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent1), #5a52e0);
    color: #fff;
    font-size: 0.95rem; font-weight: 600;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    transition: all var(--transition);
    box-shadow: 0 4px 24px rgba(108,99,255,0.3);
    letter-spacing: 0.2px;
  }
  .btn-predict:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.45);
  }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .btn-predict.loading { background: linear-gradient(135deg, #5a52e0, #4a43c0); }

  /* ── Spinner ── */
  .spinner {
    width: 18px; height: 18px;
    border: 2.5px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Results Panel ── */
  #results-panel { display: none; }

  /* ── Age Display ── */
  .age-display {
    text-align: center;
    padding: 28px 24px 20px;
    position: relative;
  }
  .age-ring {
    width: 140px; height: 140px;
    border-radius: 50%;
    background: conic-gradient(var(--accent1) var(--pct, 0%), var(--bg-glass) 0%);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 20px;
    position: relative;
    box-shadow: 0 0 40px var(--accent-glow);
  }
  .age-ring::before {
    content: '';
    position: absolute; inset: 10px;
    background: var(--bg-card);
    border-radius: 50%;
  }
  .age-ring-inner {
    position: relative; z-index: 1;
    display: flex; flex-direction: column; align-items: center;
  }
  .age-number { font-size: 2.6rem; font-weight: 800; line-height: 1; letter-spacing: -2px; }
  .age-unit { font-size: 0.72rem; color: var(--text-muted); font-weight: 500; letter-spacing: 1px; text-transform: uppercase; }

  .age-group-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 18px;
    border-radius: 100px;
    font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.3px;
  }

  /* ── Details Grid ── */
  .details-grid { display: flex; flex-direction: column; gap: 12px; }
  .detail-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 16px;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 11px;
    gap: 12px;
    transition: background var(--transition);
  }
  .detail-row:hover { background: rgba(255,255,255,0.06); }
  .detail-label { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; display: flex; align-items: center; gap: 7px; }
  .detail-value { font-size: 0.88rem; font-weight: 600; color: var(--text-primary); text-align: right; }

  /* ── Confidence Bar ── */
  .conf-bar-wrap { margin-top: 16px; }
  .conf-bar-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-muted); margin-bottom: 7px; }
  .conf-bar-bg { height: 7px; background: var(--bg-glass); border-radius: 100px; overflow: hidden; }
  .conf-bar-fill {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    width: 0%;
    transition: width 1s cubic-bezier(.4,0,.2,1);
  }

  /* ── Empty/Placeholder state ── */
  .placeholder-msg {
    text-align: center; padding: 60px 24px;
    color: var(--text-muted);
  }
  .placeholder-msg .ph-icon { font-size: 3rem; margin-bottom: 14px; opacity: 0.35; }
  .placeholder-msg p { font-size: 0.88rem; line-height: 1.6; }

  /* ── Error ── */
  .error-box {
    padding: 16px; border-radius: 12px;
    background: rgba(255,92,124,0.1);
    border: 1px solid rgba(255,92,124,0.3);
    color: #ff7a95; font-size: 0.85rem;
    margin-top: 16px; display: none;
  }

  /* ── Footer ── */
  footer {
    text-align: center; padding: 24px;
    color: var(--text-muted); font-size: 0.78rem;
    border-top: 1px solid var(--border);
  }

  /* ── Animations ── */
  .fade-in { animation: fadeIn 0.5s ease forwards; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

  .pulse-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--success);
    animation: pulse 2s infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,229,160,0.5); }
    50% { opacity: 0.8; box-shadow: 0 0 0 5px rgba(0,229,160,0); }
  }
</style>
</head>
<body>
<div class="app">

  <header>
    <div class="logo-icon">🔮</div>
    <div class="logo-text"><span>AgeVision</span> AI</div>
    <div class="header-badge">
      <span>⚡ Model Ready</span>
    </div>
  </header>

  <main>
    <div class="hero">
      <h1>Predict Age From<br>Any Portrait Photo</h1>
      <p>Upload a face image and our EfficientNet-B4 model will instantly estimate the person's age with confidence intervals.</p>
    </div>

    <div class="stats-strip">
      <div class="stat-pill"><div class="stat-dot green"></div>Trained on <strong>375,775 images</strong></div>
      <div class="stat-pill"><div class="stat-dot"></div>Validation MAE <strong>6.74 years</strong></div>
      <div class="stat-pill"><div class="stat-dot"></div>±10 yr accuracy <strong>78.1%</strong></div>
      <div class="stat-pill"><div class="stat-dot green"></div>GPU Accelerated <strong>CUDA</strong></div>
    </div>

    <div class="workspace">

      <!-- Upload Panel -->
      <div class="card">
        <div class="card-header">
          <span class="card-icon">🖼️</span>
          <span class="card-title">Upload Image</span>
        </div>
        <div class="card-body">

          <div class="upload-zone" id="upload-zone">
            <input type="file" id="file-input" accept="image/jpeg,image/png,image/webp">
            <div class="upload-icon">📸</div>
            <div class="upload-label">Drop a photo here, or click to browse</div>
            <div class="upload-sub">JPEG, PNG or WebP · Max 10MB</div>
          </div>

          <div id="preview-container">
            <img id="preview-img" src="" alt="Selected image preview">
          </div>

          <div class="error-box" id="error-box">
            ⚠️ <span id="error-msg">Something went wrong.</span>
          </div>

          <button class="btn-predict" id="predict-btn" disabled onclick="runPrediction()">
            <span id="btn-icon">🔍</span>
            <span id="btn-label">Analyse Image</span>
          </button>
        </div>
      </div>

      <!-- Results Panel -->
      <div class="card">
        <div class="card-header">
          <span class="card-icon">📊</span>
          <span class="card-title">Prediction Results</span>
        </div>
        <div class="card-body">

          <!-- Placeholder -->
          <div class="placeholder-msg" id="placeholder">
            <div class="ph-icon">🤖</div>
            <p>Upload a face photo and click<br><strong>Analyse Image</strong> to see results.</p>
          </div>

          <!-- Results content -->
          <div id="results-panel">
            <!-- Age Ring -->
            <div class="age-display">
              <div class="age-ring" id="age-ring">
                <div class="age-ring-inner">
                  <div class="age-number" id="age-number">—</div>
                  <div class="age-unit">years</div>
                </div>
              </div>
              <div id="group-badge" class="age-group-badge">—</div>
            </div>

            <!-- Details -->
            <div class="details-grid" style="margin-top: 20px;">
              <div class="detail-row">
                <span class="detail-label">🎂 Predicted Age</span>
                <span class="detail-value" id="d-age">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">📐 Likely Age Range</span>
                <span class="detail-value" id="d-range">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">👤 Age Group</span>
                <span class="detail-value" id="d-group">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">⚧ Gender</span>
                <span class="detail-value" id="d-gender">—</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">✨ Confidence</span>
                <span class="detail-value" id="d-conf">—</span>
              </div>
            </div>

            <!-- Confidence Bar -->
            <div class="conf-bar-wrap" style="margin-top: 20px; padding: 0 2px;">
              <div class="conf-bar-label">
                <span>Prediction Confidence</span>
                <span id="conf-pct-label">—</span>
              </div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" id="conf-bar"></div>
              </div>
            </div>

            <div style="margin-top: 14px; padding: 12px 14px; background: rgba(255,255,255,0.03); border-radius: 10px; font-size: 0.73rem; color: var(--text-muted); line-height: 1.6;">
              ⚠️ Model-derived confidence estimate based on validation residual statistics (±1.96 σ). Not a calibrated probability.
            </div>
          </div>

        </div>
      </div>

    </div>
  </main>

  <footer>
    AgeVision AI · EfficientNet-B4 · Trained on 375,775 images · Val MAE 6.74 years
  </footer>
</div>

<script>
const fileInput  = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const previewBox = document.getElementById('preview-container');
const uploadZone = document.getElementById('upload-zone');
const predictBtn = document.getElementById('predict-btn');
const errorBox   = document.getElementById('error-box');
const errorMsg   = document.getElementById('error-msg');

const GROUP_COLORS = {
  'Child':       { bg: 'rgba(255,186,0,0.15)', border: 'rgba(255,186,0,0.4)', text: '#ffba00' },
  'Teenager':    { bg: 'rgba(0,212,255,0.12)', border: 'rgba(0,212,255,0.35)', text: '#00d4ff' },
  'Young Adult': { bg: 'rgba(108,99,255,0.15)', border: 'rgba(108,99,255,0.4)', text: '#9b94ff' },
  'Adult':       { bg: 'rgba(0,229,160,0.12)', border: 'rgba(0,229,160,0.35)', text: '#00e5a0' },
  'Senior':      { bg: 'rgba(255,92,124,0.12)', border: 'rgba(255,92,124,0.35)', text: '#ff7a95' },
};

const GROUP_ICONS = {
  'Child': '🧒', 'Teenager': '🧑', 'Young Adult': '👩', 'Adult': '🧔', 'Senior': '👴'
};

// Drag & Drop
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) { showError('Please select an image file.'); return; }
  hideError();
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewBox.style.display = 'block';
    predictBtn.disabled = false;
    // Reset results
    document.getElementById('placeholder').style.display = 'block';
    document.getElementById('results-panel').style.display = 'none';
  };
  reader.readAsDataURL(file);
}

async function runPrediction() {
  const file = fileInput.files[0];
  if (!file) return;

  // Loading state
  predictBtn.disabled = true;
  predictBtn.classList.add('loading');
  document.getElementById('btn-icon').innerHTML = '<div class="spinner"></div>';
  document.getElementById('btn-label').textContent = 'Analysing...';
  hideError();

  const formData = new FormData();
  formData.append('image', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || 'Prediction failed. Please try another image.');
      return;
    }

    displayResults(data);

  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.classList.remove('loading');
    document.getElementById('btn-icon').textContent = '🔍';
    document.getElementById('btn-label').textContent = 'Analyse Image';
    predictBtn.disabled = false;
  }
}

function displayResults(data) {
  document.getElementById('placeholder').style.display = 'none';
  const panel = document.getElementById('results-panel');
  panel.style.display = 'block';
  panel.classList.add('fade-in');

  const age    = data.predicted_age;
  const group  = data.predicted_age_group;
  const range  = data.likely_age_range;
  const conf   = data.confidence_level;
  const gender = data.predicted_gender;
  const gconf  = data.gender_confidence;

  // Extract confidence percentage from string e.g. "Model-derived: High (81.7%)"
  const confMatch = conf.match(/([\d.]+)%/);
  const confPct   = confMatch ? parseFloat(confMatch[1]) : 70;
  const confLabel = confMatch ? conf.split('(')[0].replace('Model-derived: ', '').trim() : conf;

  // Age ring
  const pct = Math.min(age / 100, 1);
  document.getElementById('age-ring').style.setProperty('--pct', (pct * 360) + 'deg');
  animateNumber('age-number', 0, age, 900);

  // Group badge
  const colors = GROUP_COLORS[group] || GROUP_COLORS['Young Adult'];
  const icon   = GROUP_ICONS[group] || '👤';
  const badge  = document.getElementById('group-badge');
  badge.textContent = `${icon} ${group}`;
  badge.style.background  = colors.bg;
  badge.style.border      = `1px solid ${colors.border}`;
  badge.style.color       = colors.text;

  // Details
  document.getElementById('d-age').textContent   = `${age} years`;
  document.getElementById('d-range').textContent = range;
  document.getElementById('d-group').textContent = group;
  document.getElementById('d-gender').textContent = `${gender} (${gconf})`;
  document.getElementById('d-conf').textContent  = confLabel;

  // Confidence bar
  document.getElementById('conf-pct-label').textContent = confPct.toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('conf-bar').style.width = confPct + '%';
  }, 100);
}

function animateNumber(id, from, to, duration) {
  const el = document.getElementById(id);
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (to - from) * ease).toFixed(1);
    if (t < 1) requestAnimationFrame(update);
    else el.textContent = to.toFixed(1);
  };
  requestAnimationFrame(update);
}

function showError(msg) { errorMsg.textContent = msg; errorBox.style.display = 'block'; }
function hideError() { errorBox.style.display = 'none'; }
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Read uploaded image as PIL and convert to RGB NumPy array
        image_bytes = file.read()
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
        img_array = np.array(pil_image)

        # Gunicorn imports app.py instead of executing the
        # __main__ block, so initialize the predictor here.
        load_predictor()

        # Safety check
        if predictor is None:
            return jsonify({
                "error": "Age prediction model failed to initialize."
            }), 500

        # Run prediction
        result = predictor.predict(img_array)

        # Handle model-level errors
        if "error" in result:
            return jsonify({
                "error": result["error"]
            }), 400

        # Return prediction results
        return jsonify({
            "predicted_age": result["predicted_age"],
            "predicted_age_group": result["predicted_age_group"],
            "likely_age_range": result["likely_age_range"],
            "confidence_level": result["confidence_level"],
            "predicted_gender": result.get(
                "predicted_gender",
                "Unknown"
            ),
            "gender_confidence": result.get(
                "gender_confidence",
                "0%"
            )
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  AgeVision AI — Starting Web Server")
    print("=" * 60)
    load_predictor()
    print("\n  Open your browser: http://127.0.0.1:8080\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
