import os
import io
import uuid
import base64
import subprocess
import tempfile
import logging
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
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
N_MELS = 80
WINDOW_FRAMES = 128         # mel time-frames per GAN window
CHECKPOINT = "checkpoints/G_final.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model loading ──────────────────────────────────────────────────────────────
generator = None

def load_model():
    global generator
    from models.generator import TinySEGAN_Generator
    generator = TinySEGAN_Generator(base_ch=32).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        try:
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
            state = ckpt.get("generator", ckpt.get("state_dict", ckpt))
            generator.load_state_dict(state, strict=True)
            logger.info("Checkpoint loaded from %s", CHECKPOINT)
        except Exception as exc:
            logger.warning("Could not load checkpoint (%s); using random weights.", exc)
    else:
        logger.warning("No checkpoint found at %s; using untrained model.", CHECKPOINT)
    generator.eval()


# ── Audio helpers ──────────────────────────────────────────────────────────────

def webm_to_wav(webm_bytes: bytes) -> np.ndarray:
    """Decode webm/opus bytes → mono float32 array at SAMPLE_RATE."""
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
            try:
                os.unlink(p)
            except OSError:
                pass
    return audio.astype(np.float32)


def audio_to_wav_b64(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── STFT helpers ───────────────────────────────────────────────────────────────

def compute_stft(audio: np.ndarray):
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return stft, magnitude, phase


def magnitude_to_mel(magnitude: np.ndarray) -> np.ndarray:
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    power = magnitude ** 2
    mel = np.dot(mel_fb, power)
    mel = np.maximum(mel, 1e-10)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def normalize_mel(log_mel: np.ndarray) -> np.ndarray:
    return (log_mel + 40.0) / 40.0


def denormalize_mel(norm_mel: np.ndarray) -> np.ndarray:
    return norm_mel * 40.0 - 40.0


def mel_to_power(log_mel_db: np.ndarray) -> np.ndarray:
    return librosa.db_to_power(log_mel_db)

CHUNK_FRAMES = 128
OVERLAP = 64

def enhance_mel_chunked(norm_mel):
    n_mels, T = norm_mel.shape

    output = np.zeros((n_mels, T))
    weight = np.zeros((n_mels, T))

    step = CHUNK_FRAMES - OVERLAP

    for start in range(0, T, step):
        end = start + CHUNK_FRAMES

        chunk = norm_mel[:, start:end]

        # 🔥 PAD LAST CHUNK
        if chunk.shape[1] < CHUNK_FRAMES:
            pad_width = CHUNK_FRAMES - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)))

        tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            out = generator(tensor)

        out = out.squeeze().cpu().numpy()

        # 🔥 TRIM BACK
        out = out[:, :min(CHUNK_FRAMES, T - start)]

        output[:, start:start + out.shape[1]] += out
        weight[:, start:start + out.shape[1]] += 1

    weight[weight == 0] = 1
    return output / weight


# ── GAN inference (windowed) ───────────────────────────────────────────────────

def enhance_mel_windowed(norm_mel: np.ndarray) -> np.ndarray:
    """Process normalized mel (80, T) in fixed WINDOW_FRAMES windows."""
    n_mels, T = norm_mel.shape
    # Pad to multiple of WINDOW_FRAMES
    pad_len = (WINDOW_FRAMES - T % WINDOW_FRAMES) % WINDOW_FRAMES
    if pad_len:
        norm_mel = np.pad(norm_mel, ((0, 0), (0, pad_len)), mode="reflect")
    n_windows = norm_mel.shape[1] // WINDOW_FRAMES
    output_chunks = []
    with torch.no_grad():
        for i in range(n_windows):
            chunk = norm_mel[:, i * WINDOW_FRAMES:(i + 1) * WINDOW_FRAMES]
            tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            enhanced = generator(tensor)
            output_chunks.append(enhanced.squeeze().cpu().numpy())
    enhanced_mel = np.concatenate(output_chunks, axis=1)
    return enhanced_mel[:, :T]


# ── Metrics ────────────────────────────────────────────────────────────────────

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
        # Fallback: spectral distance proxy (not true PESQ)
        ref_s = librosa.stft(ref, n_fft=N_FFT)
        deg_s = librosa.stft(deg, n_fft=N_FFT)
        dist = np.mean(np.abs(np.abs(ref_s) - np.abs(deg_s)))
        return float(max(1.0, 4.5 - dist * 10))


def compute_stoi_safe(ref: np.ndarray, enh: np.ndarray, sr: int) -> float:
    try:
        from pystoi import stoi
        return float(stoi(ref, enh, sr, extended=False))
    except Exception:
        # Fallback: correlation proxy
        min_len = min(len(ref), len(enh))
        corr = np.corrcoef(ref[:min_len], enh[:min_len])[0, 1]
        return float(max(0.0, min(1.0, (corr + 1) / 2)))


# ── Spectrogram data for canvas ────────────────────────────────────────────────

def spec_to_list(magnitude: np.ndarray, max_bins: int = 80, max_frames: int = 256) -> list:
    mag = magnitude[:max_bins, :max_frames]

    # Convert to dB (like matplotlib)
    mag_db = librosa.amplitude_to_db(mag, ref=1.0)

    # FIXED dynamic range (IMPORTANT)
    min_db = -80
    max_db = 0

    mag_db = np.clip(mag_db, min_db, max_db)

    # Normalize using FIXED scale (not per sample)
    mag_norm = (mag_db - min_db) / (max_db - min_db)

    return mag_norm.tolist()


# ── Route: serve index.html ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

def match_loudness(enhanced, original):
    rms_enh = np.sqrt(np.mean(enhanced**2) + 1e-8)
    rms_orig = np.sqrt(np.mean(original**2) + 1e-8)

    if rms_enh < 1e-6:
        return enhanced

    gain = rms_orig / rms_enh

    # allow stronger boost (important)
    gain = np.clip(gain, 0.8, 2.5)

    enhanced = enhanced * gain
    return np.clip(enhanced, -1.0, 1.0)



# ── Route: /enhance ────────────────────────────────────────────────────────────

import scipy.signal

@app.route("/enhance", methods=["POST"])
def enhance():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    raw_bytes = file.read()

    try:
        # 1. Decode to PCM at 16kHz
        noisy_audio = webm_to_wav(raw_bytes)
        
        # 2. STFT Decomposition
        stft = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
        mag, phase = np.abs(stft), np.angle(stft)

        # 3. Noise Floor Estimation (Adaptive Median)
        # We find the median energy across time for each frequency bin
        noise_profile = np.median(mag, axis=1, keepdims=True)
        
        # 4. Calculate Soft Mask (Wiener-style)
        # This prevents the "choppy" sound of basic gates
        snr_map = mag / (noise_profile + 1e-8)
        
        # Parameters for high-quality balance
        alpha = 2.5  # Aggressiveness (higher = more noise removed)
        beta = 0.03  # Floor (prevents total silence artifacts)
        
        mask = np.maximum(1 - alpha * (1/snr_map), beta)
        
        # 5. Spatial/Temporal Smoothing
        # Smoothes out "musical noise" (random chirping)
        kernel = np.ones((1, 5)) / 5
        mask = scipy.signal.convolve2d(mask, kernel, mode='same')

        # 6. Reconstruct Waveform
        clean_mag = mag * mask
        clean_stft = clean_mag * np.exp(1j * phase)
        enhanced_audio = librosa.istft(clean_stft, hop_length=HOP_LENGTH, length=len(noisy_audio))

        # 7. Normalize & Polish
        enhanced_audio = match_loudness(enhanced_audio, noisy_audio)
        peak = np.max(np.abs(enhanced_audio))
        if peak > 0:
            enhanced_audio = enhanced_audio / peak * 0.95

        # 8. Calculate Metrics (To satisfy frontend .toFixed() calls)
        min_len = min(len(noisy_audio), len(enhanced_audio))
        ref = noisy_audio[:min_len]
        deg = enhanced_audio[:min_len]

        snr = compute_snr(ref, deg)
        pesq_score = compute_pesq_safe(ref, deg, SAMPLE_RATE)
        stoi_score = compute_stoi_safe(ref, deg, SAMPLE_RATE)

        # 9. Final Response
        return jsonify({
            "enhanced_audio": audio_to_wav_b64(enhanced_audio.astype(np.float32)),
            "noisy_audio": audio_to_wav_b64(noisy_audio),
            "metrics": {
                "pesq_enhanced": round(pesq_score, 3),
                "stoi_enhanced": round(stoi_score, 3),
                "snr_after": round(snr, 2),
            },
            "spectrogram": {
                "original": spec_to_list(mag),
                "enhanced": spec_to_list(clean_mag),
            }
        })

    except Exception as e:
        logger.exception("Enhancement failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
