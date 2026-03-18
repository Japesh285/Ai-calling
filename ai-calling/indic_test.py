import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Correct repo ID (IMPORTANT)
repo_id = "ai4bharat/IndicF5"

print("Loading model...")
model = AutoModel.from_pretrained(
    repo_id,
    trust_remote_code=True
).to(device)

print("Model loaded.")

# Your input text
text = "नमस्ते आपका स्वागत है"

# VERY IMPORTANT: you NEED a reference audio
ref_audio_path = "prompt.wav"   # <-- you must provide this
ref_text = "नमस्ते आप कैसे हैं"

print("Generating audio...")

with torch.inference_mode():
    audio = model(
        text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text
    )

# Normalize if needed
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

# Save
sf.write("output.wav", np.array(audio, dtype=np.float32), 24000)

print("Done! Audio saved.")