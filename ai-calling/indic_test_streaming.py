"""
Simple test to check if Indic Parler TTS supports streaming with ParlerTTSStreamer
"""
import torch
import numpy as np
import soundfile as sf
import os
import time
from threading import Thread

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_REPO = "ai4bharat/indic-parler-tts"

# Try to import ParlerTTSStreamer
try:
    from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
    from transformers import AutoTokenizer
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    
    # Register parler_tts config
    if 'parler_tts' not in CONFIG_MAPPING:
        CONFIG_MAPPING.register('parler_tts', ParlerTTSForConditionalGeneration, exist_ok=True)
    
    print("✓ ParlerTTSStreamer import successful")
    STREAMING_AVAILABLE = True
except ImportError as e:
    print(f"✗ ParlerTTSStreamer import failed: {e}")
    STREAMING_AVAILABLE = False
    exit(1)

print(f"\nLoading model: {MODEL_REPO}...")

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    local_files_only=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    local_files_only=True
)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path,
    trust_remote_code=True,
    local_files_only=True
)

print("✓ Model loaded\n")

# Test text
text = "नमस्ते आपका स्वागत है"
description = "A female speaker delivers a clear and friendly speech."

print("="*50)
print("TESTING STREAMING GENERATION")
print("="*50)
print(f"Text: {text}")

# Setup inputs
description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)
max_length = 500

# Create streamer
print("\nCreating ParlerTTSStreamer...")
streamer = ParlerTTSStreamer(
    model,
    device=device,
    play_steps=50,
    stride=5,
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

# Collect chunks
chunks = []
chunk_count = 0
first_chunk_time = None
start_time = time.monotonic()

def run_generation():
    with torch.inference_mode():
        model.generate(**generation_kwargs)
    streamer.end()

print("Starting generation thread...")
thread = Thread(target=run_generation)
thread.start()

print("Waiting for chunks...\n")
try:
    for chunk in streamer:
        elapsed = time.monotonic() - start_time
        chunk_count += 1
        chunks.append(chunk)
        
        if first_chunk_time is None:
            first_chunk_time = elapsed
            print(f"  [✓] First chunk: {elapsed:.2f}s ({len(chunk)} samples)")
        else:
            print(f"  [✓] Chunk {chunk_count}: {elapsed:.2f}s ({len(chunk)} samples)")
except Exception as e:
    print(f"  [✗] Error: {e}")

thread.join(timeout=10.0)
total_time = time.monotonic() - start_time

print(f"\n" + "="*50)
print("RESULTS")
print("="*50)

if chunks:
    print(f"✓ STREAMING WORKS!")
    print(f"  Chunks received: {chunk_count}")
    print(f"  Time to first chunk: {first_chunk_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    
    # Save audio
    audio = np.concatenate(chunks)
    output_path = "output_streaming_test.wav"
    sample_rate = getattr(model.config, 'sampling_rate', 24000)
    sf.write(output_path, audio.astype(np.float32), sample_rate)
    print(f"  Audio saved to: {output_path}")
    print(f"  Duration: {len(audio) / sample_rate:.2f}s")
else:
    print(f"✗ STREAMING FAILED - No chunks received")

print("="*50)
