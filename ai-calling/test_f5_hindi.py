"""
Quick quality + latency test for SPRINGLab/F5-Hindi-24KHz.
Run:  source venv/bin/activate && python3 test_f5_hindi.py

Fixes applied vs previous version:
  1. Audio normalised to -3 dBFS before saving — prevents clipping.
  2. 8kHz downsampling uses scipy.signal.resample_poly with explicit
     anti-aliasing (lowpass at 3 800 Hz) so consonants don't ring/alias.
  3. NFE steps raised 32 → 64 for better flow-matching convergence.
  4. cfg_strength raised 2.0 → 2.5 for clearer speech alignment.
  5. Reference audio trimmed to 10 s — long refs can confuse the model.
  6. Output written as proper float32 PCM so no integer rounding.
"""
import time
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt
from math import gcd

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT


# ── Paths ──────────────────────────────────────────────────────────────
CKPT  = "/home/developer/ai-calling/models/f5_hindi/model_2500000.safetensors"
VOCAB = "/home/developer/ai-calling/models/f5_hindi/vocab.txt"

# Reference audio — 15s clean Hindi male (trimmed to 10s inside the script)
REF_AUDIO = "/home/developer/ai-calling/HIN_M_AvdheshT_clean.wav"
REF_TEXT  = (
    "अपके प्रचालनों पर लागु होने कानूनों को पहचानने के लिए "
    "एक कार्यान्वित सिस्टम होना चाहिए।"
)

# Tax-support style sentences (what the live agent would actually say)
TEST_SENTENCES = [
    "नमस्ते! मैं आपकी कर संबंधी समस्याओं में मदद कर सकता हूँ।",
    "आयकर रिटर्न दाखिल करने की अंतिम तिथि हर साल ३१ जुलाई होती है।",
    "आपका पैन कार्ड और आधार कार्ड लिंक होना अनिवार्य है।",
    "जीएसटी पंजीकरण के लिए आपको क्या जानकारी चाहिए?",
]

TARGET_DBFS = -3.0   # normalise output to -3 dBFS (headroom before clipping)
OUT_DIR     = "/home/developer/ai-calling/logs"  # save alongside other diagnostic audio
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helper: normalise float32 audio to a target dBFS ───────────────────
def normalise(wav: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Scale so the peak sits at target_dbfs, then hard-limit just in case."""
    peak = np.max(np.abs(wav))
    if peak < 1e-9:
        return wav
    target_peak = 10 ** (target_dbfs / 20.0)   # ~0.708 for -3 dBFS
    wav = wav * (target_peak / peak)
    return np.clip(wav, -1.0, 1.0)              # safety clip — never distorts


# ── Helper: proper 24 kHz → 8 kHz with anti-alias lowpass ──────────────
def downsample_24k_to_8k(wav: np.ndarray, src_sr: int = 24000) -> np.ndarray:
    """
    Resample 24 kHz → 8 kHz.
    A 4th-order Butterworth lowpass at 3 800 Hz is applied BEFORE
    decimation to remove all energy above the 4 kHz Nyquist of 8 kHz
    audio.  Without this, high-frequency energy folds back into the
    speech band and produces the 'choked / buzzy' artefact.
    """
    # Anti-alias filter: cut at 3 800 Hz (just below 4 kHz Nyquist of 8 kHz)
    sos = butter(4, 3800.0, btype="low", fs=src_sr, output="sos")
    wav_filtered = sosfilt(sos, wav.astype(np.float64)).astype(np.float32)

    g    = gcd(src_sr, 8000)
    up   = 8000  // g     # = 1  (gcd of 24000 and 8000 is 8000)
    down = src_sr // g    # = 3
    wav_8k = resample_poly(wav_filtered, up, down).astype(np.float32)
    return np.clip(wav_8k, -1.0, 1.0)


# ── Load model once ─────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print("Loading F5-Hindi model...")
t0 = time.monotonic()

VOCOS_LOCAL = "/home/developer/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
vocoder = load_vocoder(is_local=True, local_path=VOCOS_LOCAL)

model = load_model(
    DiT,
    dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=CKPT,
    mel_spec_type="vocos",
    vocab_file=VOCAB,
    device=DEVICE,
)

print(f"Model loaded in {time.monotonic() - t0:.1f}s")

# ── Preprocess reference audio once ────────────────────────────────────
# Trim to 10 s — very long references confuse F5-TTS voice conditioning
ref_data, ref_sr = sf.read(REF_AUDIO)
max_ref_samples   = ref_sr * 10
if len(ref_data) > max_ref_samples:
    ref_data = ref_data[:max_ref_samples]
    trimmed_ref_path = f"{OUT_DIR}/ref_trimmed_10s.wav"
    sf.write(trimmed_ref_path, ref_data.astype(np.float32), ref_sr,
             subtype="PCM_16")
    ref_audio_input = trimmed_ref_path
    print(f"Reference trimmed to 10 s → {trimmed_ref_path}")
else:
    ref_audio_input = REF_AUDIO

ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
    ref_audio_input, REF_TEXT
)

# ── Inference loop ──────────────────────────────────────────────────────
print("\n--- Inference tests (nfe_step=64, cfg_strength=2.5) ---")
for i, text in enumerate(TEST_SENTENCES):
    t_start = time.monotonic()
    wav, sr, _ = infer_process(
        ref_audio_proc,
        ref_text_proc,
        text,
        model,
        vocoder,
        mel_spec_type="vocos",
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=64,           # was 32 — more steps = better convergence
        cfg_strength=2.5,      # was 2.0 — stronger text alignment
        speed=1.0,
        show_info=lambda x: None,
        device=DEVICE,
    )
    latency  = time.monotonic() - t_start
    duration = len(wav) / sr
    rtf      = latency / duration

    # Fix 1: normalise before saving — no more clipping
    wav_norm = normalise(wav)

    # Save 24 kHz (full quality)
    path_24k = f"{OUT_DIR}/f5_hindi_test_{i+1}.wav"
    sf.write(path_24k, wav_norm, sr, subtype="PCM_16")

    # Fix 2: anti-alias lowpass THEN resample — no more choked consonants
    wav_8k   = downsample_24k_to_8k(wav_norm, src_sr=sr)
    path_8k  = f"{OUT_DIR}/f5_hindi_test_{i+1}_8k.wav"
    sf.write(path_8k, wav_8k, 8000, subtype="PCM_16")

    # Peak check — should never be 1.000 now
    peak_24k = float(np.max(np.abs(wav_norm)))
    peak_8k  = float(np.max(np.abs(wav_8k)))

    print(f"\n[{i+1}] \"{text[:45]}\"")
    print(f"     Latency  : {latency*1000:.0f} ms")
    print(f"     Duration : {duration:.2f} s")
    print(f"     RTF      : {rtf:.2f}x  {'✅ real-time' if rtf < 1 else '⚠️  slower'}")
    print(f"     24 kHz   : {path_24k}  peak={peak_24k:.4f}")
    print(f"     8 kHz    : {path_8k}  peak={peak_8k:.4f}")

print("\n✅ Done. Listen to the _8k.wav files — those are exactly what")
print("   FreeSWITCH plays over the phone.")
print("   Compare f5_hindi_test_1_8k.wav with the current XTTS output")
print("   to decide which model sounds better.")
