import os
import io
import base64
import subprocess
import tempfile
import logging
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128

# ── Audio helpers ──────────────────────────────────────────────────────────────

def webm_to_wav(webm_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(webm_bytes)
        src_path = src.name
    wav_path = src_path.replace(".webm", ".wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "wav",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()}")
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    finally:
        for p in (src_path, wav_path):
            if os.path.exists(p):
                os.unlink(p)
    return audio.astype(np.float32)

def audio_to_wav_b64(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Metrics & Helpers ─────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2) + 1e-10
    return float(10 * np.log10(signal_power / noise_power))

def compute_pesq_safe(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    try:
        from pesq import pesq
        mode = "wb" if sr >= 16000 else "nb"
        return float(pesq(sr, ref, deg, mode))
    except Exception:
        return 1.0

def compute_stoi_safe(ref: np.ndarray, enh: np.ndarray, sr: int) -> float:
    try:
        from pystoi import stoi
        return float(stoi(ref, enh, sr, extended=False))
    except Exception:
        return 0.5

def spec_to_list(magnitude: np.ndarray, max_bins: int = 80, max_frames: int = 256) -> list:
    mag = magnitude[:max_bins, :max_frames]
    mag_db = librosa.amplitude_to_db(mag, ref=1.0)
    min_db, max_db = -80, 0
    mag_db = np.clip(mag_db, min_db, max_db)
    mag_norm = (mag_db - min_db) / (max_db - min_db)
    return mag_norm.tolist()

def match_loudness(enhanced, original):
    rms_enh = np.sqrt(np.mean(enhanced**2) + 1e-8)
    rms_orig = np.sqrt(np.mean(original**2) + 1e-8)
    gain = np.clip(rms_orig / (rms_enh + 1e-8), 0.8, 2.5)
    return np.clip(enhanced * gain, -1.0, 1.0)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/enhance", methods=["POST"])
def enhance():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    raw_bytes = file.read()

    try:
        noisy_audio = webm_to_wav(raw_bytes)
        
        # 1. STFT
        stft = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
        mag, phase = np.abs(stft), np.angle(stft)

        # 2. Spectral Gating (High Quality Non-GAN)
        noise_profile = np.median(mag, axis=1, keepdims=True)
        snr_map = mag / (noise_profile + 1e-8)
        
        alpha, beta = 2.5, 0.03
        mask = np.maximum(1 - alpha * (1/snr_map), beta)
        
        # Smooth mask to prevent artifacts
        kernel = np.ones((1, 5)) / 5
        mask = scipy.signal.convolve2d(mask, kernel, mode='same')

        # 3. Reconstruct
        clean_mag = mag * mask
        clean_stft = clean_mag * np.exp(1j * phase)
        enhanced_audio = librosa.istft(clean_stft, hop_length=HOP_LENGTH, length=len(noisy_audio))

        # 4. Final Polish
        enhanced_audio = match_loudness(enhanced_audio, noisy_audio)
        peak = np.max(np.abs(enhanced_audio))
        if peak > 0:
            enhanced_audio = enhanced_audio / peak * 0.95

        # 5. Metrics
        min_len = min(len(noisy_audio), len(enhanced_audio))
        ref, deg = noisy_audio[:min_len], enhanced_audio[:min_len]
        
        res_json = {
            "enhanced_audio": audio_to_wav_b64(enhanced_audio.astype(np.float32)),
            "noisy_audio": audio_to_wav_b64(noisy_audio),
            "metrics": {
                "pesq_enhanced": round(compute_pesq_safe(ref, deg, SAMPLE_RATE), 3),
                "stoi_enhanced": round(compute_stoi_safe(ref, deg, SAMPLE_RATE), 3),
                "snr_after": round(compute_snr(ref, deg), 2),
            },
            "spectrogram": {
                "original": spec_to_list(mag),
                "enhanced": spec_to_list(clean_mag),
            }
        }
        return jsonify(res_json)

    except Exception as e:
        logger.exception("Enhancement failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Remove load_model() call to prevent crash on missing folders
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
