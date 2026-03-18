import torch
import soundfile as sf
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

# 🔥 FIX for PyTorch 2.6+
torch.serialization.add_safe_globals([XttsConfig])

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
REFERENCE_AUDIO = "/home/developer/ai-calling/abhishek_reference.wav"
OUTPUT_FILE = "output_hindi.wav"

TEXT = "Namaste! Mera naam Japesh hai. Main aapki kaise madad kar sakta hoon?"

print("🚀 Loading XTTS model...")
tts = TTS(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)

print(f"✅ Model loaded on {device}")

wav = tts.tts(
    text=TEXT,
    speaker_wav=REFERENCE_AUDIO,
    language="hi"
)

sf.write(OUTPUT_FILE, wav, samplerate=24000)

print("✅ Done! Check output_hindi.wav")