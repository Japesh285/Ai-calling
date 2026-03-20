"""
Indic TTS test script for Parler TTS (AI Bharat)
Supports multiple Indic languages with Parler-TTS architecture.
Uses locally cached model: ai4bharat/indic-parler-tts

TESTS:
1. Blocking generation (standard)
2. Streaming generation (ParlerTTSStreamer)
"""
import torch
import numpy as np
import soundfile as sf
import os
import time
from threading import Thread

# Try to import streaming support
try:
    from parler_tts import ParlerTTSStreamer
    STREAMING_AVAILABLE = True
    print("✓ ParlerTTSStreamer available")
except ImportError:
    STREAMING_AVAILABLE = False
    print("✗ ParlerTTSStreamer NOT available")

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


def generate_blocking(text, description, model, tokenizer, description_tokenizer, ref_inputs, device):
    """Standard blocking generation - waits for complete audio."""
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)
    
    estimated_audio_tokens = prompt_input_ids.input_ids.shape[1] * 15
    max_length = min(estimated_audio_tokens + 50, 1500)
    
    start_time = time.monotonic()
    
    with torch.inference_mode():
        if ref_inputs is not None:
            generation = model.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask,
                **ref_inputs,
                pad_token_id=tokenizer.pad_token_id,
                max_length=max_length,
                min_new_tokens=10,
                do_sample=True,
                temperature=1.0,
            )
        else:
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
        
        if hasattr(generation, 'audio_values'):
            audio = generation.audio_values.cpu().numpy()[0]
        else:
            audio = generation.cpu().numpy()[0]
    
    elapsed = time.monotonic() - start_time
    return audio, elapsed


def test_streaming(text, description, model, tokenizer, description_tokenizer, ref_inputs, device, sample_rate):
    """Test streaming generation with ParlerTTSStreamer."""
    if not STREAMING_AVAILABLE:
        print("\n=== STREAMING TEST SKIPPED ===")
        print("ParlerTTSStreamer not available")
        return None, None
    
    print("\n" + "="*60)
    print("TEST 2: STREAMING GENERATION")
    print("="*60)
    
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)
    estimated_audio_tokens = prompt_input_ids.input_ids.shape[1] * 15
    max_length = min(estimated_audio_tokens + 50, 1500)
    
    # Create streamer - play_steps determines chunk size
    # play_steps=50 ≈ 0.55 seconds of audio per chunk at 44.1kHz
    play_steps = 50
    print(f"Creating ParlerTTSStreamer with play_steps={play_steps}...")
    
    streamer = ParlerTTSStreamer(
        model,
        device=device,
        play_steps=play_steps,
        stride=5,  # Overlap for smoother playback
    )
    
    generation_kwargs = dict(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
        min_new_tokens=10,
        do_sample=True,
        temperature=1.0,
    )
    
    if ref_inputs is not None:
        generation_kwargs['ref_inputs'] = ref_inputs
    
    # Collect chunks and timing
    chunks = []
    chunk_times = []
    first_chunk_time = None
    generation_start = time.monotonic()
    
    def run_generation():
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as e:
            print(f"Generation error: {e}")
        finally:
            streamer.end()
    
    # Start generation in background thread
    print("Starting generation in background thread...")
    thread = Thread(target=run_generation)
    thread.start()
    
    # Collect chunks as they arrive
    print("Collecting audio chunks...")
    try:
        for chunk in streamer:
            chunk_time = time.monotonic() - generation_start
            chunk_times.append(chunk_time)
            chunks.append(chunk)
            
            if first_chunk_time is None:
                first_chunk_time = chunk_time
                print(f"  ✓ First chunk received at {first_chunk_time:.2f}s ({len(chunk)} samples)")
            else:
                print(f"  ✓ Chunk {len(chunks)} received at {chunk_time:.2f}s ({len(chunk)} samples)")
    except Exception as e:
        print(f"Stream iteration error: {e}")
    
    thread.join(timeout=5.0)
    total_time = time.monotonic() - generation_start
    
    print(f"\n--- Streaming Results ---")
    print(f"  Chunks received: {len(chunks)}")
    print(f"  Time to first chunk: {first_chunk_time:.2f}s" if first_chunk_time else "  Time to first chunk: N/A")
    print(f"  Total generation time: {total_time:.2f}s")
    
    if chunks:
        # Concatenate chunks
        audio = np.concatenate(chunks)
        audio_duration = len(audio) / sample_rate
        print(f"  Total audio samples: {len(audio)}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        
        # Calculate latency improvement
        if first_chunk_time:
            latency_ratio = first_chunk_time / total_time if total_time > 0 else 0
            print(f"  Latency improvement: {(1 - latency_ratio) * 100:.1f}% faster first audio")
        
        return audio, total_time
    
    return None, None


def test_blocking(text, description, model, tokenizer, description_tokenizer, ref_inputs, device, sample_rate):
    """Test standard blocking generation."""
    print("\n" + "="*60)
    print("TEST 1: BLOCKING GENERATION")
    print("="*60)
    
    audio, elapsed = generate_blocking(
        text, description, model, tokenizer, description_tokenizer, ref_inputs, device
    )
    
    audio_duration = len(audio) / sample_rate
    print(f"\n--- Blocking Results ---")
    print(f"  Total generation time: {elapsed:.2f}s")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Real-time factor: {elapsed / audio_duration:.2f}x")
    
    return audio, elapsed

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