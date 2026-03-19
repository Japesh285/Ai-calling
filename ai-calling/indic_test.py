"""
Indic TTS test script for Parler TTS (AI Bharat)
Supports multiple Indic languages with Parler-TTS architecture.
Uses locally cached model: ai4bharat/indic-parler-tts
"""
import torch
import numpy as np
import soundfile as sf
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use locally cached AI Bharat Indic Parler TTS model
MODEL_REPO = "ai4bharat/indic-parler-tts"

# Language mapping for Indic languages
INDIC_LANGUAGES = {
    "hi": "hindi",
    "bn": "bengali",
    "te": "telugu",
    "mr": "marathi",
    "ta": "tamil",
    "gu": "gujarati",
    "kn": "kannada",
    "pa": "punjabi",
    "ml": "malayalam",
    "or": "odia",
}

print(f"Loading model: {MODEL_REPO}...")

# Import parler_tts and register config with transformers
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Manually register parler_tts config (required for Auto classes to work)
if 'parler_tts' not in CONFIG_MAPPING:
    CONFIG_MAPPING.register('parler_tts', ParlerTTSConfig, exist_ok=True)

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    local_files_only=True  # Use cached model only
).to(device)

# Two tokenizers are required:
# 1. Prompt tokenizer (for text to synthesize) - uses model's LlamaTokenizer
# 2. Description tokenizer (for voice characteristics) - uses T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    local_files_only=True
)
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path,  # google/flan-t5-large
    trust_remote_code=True,
    local_files_only=True
)

print("Model loaded.")

# Input configuration
text = "नमस्ते आपका स्वागत है"
language = "hi"  # Hindi (change as needed for other Indic languages)

# Voice description - controls speaker characteristics
description = "A female speaker delivers a clear and friendly speech with moderate speed and pitch. The recording is of high quality, with the speaker's voice sounding natural and warm."

# For Parler TTS, you can optionally use a reference audio for voice cloning
# If not available, the model will use its default voice
ref_audio_path = "prompt.wav"  # Optional: reference audio for voice cloning

# Check if reference audio exists
use_reference = os.path.exists(ref_audio_path)

if use_reference:
    print(f"Using reference audio: {ref_audio_path}")
    from transformers import AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        local_files_only=True
    )

    import librosa
    ref_audio, ref_sr = librosa.load(ref_audio_path, sr=feature_extractor.sampling_rate)

    ref_inputs = feature_extractor(
        ref_audio,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt"
    ).to(device)
else:
    print("No reference audio found, using default voice")
    ref_inputs = None

print("Generating audio...")

# Tokenize description (voice characteristics) - goes to text encoder
description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)

# Tokenize prompt (text to synthesize) - goes to prompt encoder
prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)

# Calculate appropriate max_length based on input text length
# Rough estimate: ~15 audio tokens per text token for 44kHz audio
estimated_audio_tokens = prompt_input_ids.input_ids.shape[1] * 15
max_length = min(estimated_audio_tokens + 50, 1500)  # Cap at 1500 tokens

print(f"Generating audio (max_length: {max_length})...")

# Generate audio with Parler TTS
with torch.inference_mode():
    if use_reference and ref_inputs is not None:
        # Voice cloning mode
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
            **ref_inputs,
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_length,
            min_new_tokens=10,  # Ensure some audio is generated
            do_sample=True,
            temperature=1.0,
        )
    else:
        # Standard TTS mode with better generation parameters
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_length,
            min_new_tokens=10,
            do_sample=True,
            temperature=1.0,
        )

    # Convert to numpy - generate returns tensor directly in some versions
    if hasattr(generation, 'audio_values'):
        audio = generation.audio_values.cpu().numpy()[0]
    else:
        audio = generation.cpu().numpy()[0]

# Trim silence/noise from the end of audio
def trim_silence(wav, threshold=0.01, min_silence_samples=4410):
    """Trim trailing silence/noise from audio."""
    # Find the last significant audio sample
    rms = np.sqrt(np.mean(wav**2))
    effective_threshold = max(threshold, rms * 0.1)
    
    # Scan from end to find where audio drops below threshold
    silence_count = 0
    end_idx = len(wav)
    for i in range(len(wav) - 1, -1, -1):
        if abs(wav[i]) < effective_threshold:
            silence_count += 1
            if silence_count > min_silence_samples:
                end_idx = i + min_silence_samples
                break
        else:
            silence_count = 0
    
    return wav[:end_idx]

# Trim trailing noise (common with Parler TTS)
audio = trim_silence(audio, threshold=0.02, min_silence_samples=8820)  # 0.2 seconds at 44.1kHz

# Normalize audio to prevent clipping
def normalize_audio(wav, target_dbfs=-3.0):
    """Normalize audio to target dBFS level."""
    peak = np.max(np.abs(wav))
    if peak < 1e-9:
        return wav
    target_peak = 10 ** (target_dbfs / 20.0)
    wav = wav * (target_peak / peak)
    return np.clip(wav, -1.0, 1.0)

audio = normalize_audio(audio)

# Get sampling rate from model config
sample_rate = getattr(model.config, 'sampling_rate', 24000)

# Save output
output_path = "output_parler.wav"
sf.write(output_path, audio.astype(np.float32), sample_rate)

print(f"Done! Audio saved to: {output_path}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")