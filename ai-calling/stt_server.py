import torch
from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

print("Loading Whisper model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(
    "medium",
    device=device,
    compute_type=compute_type,
)

print(f"Whisper model loaded on {device} ({compute_type})")


@app.post("/transcribe")
async def transcribe(audio: UploadFile):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(await audio.read())
        path = f.name

    segments, info = model.transcribe(path)

    text = " ".join([seg.text for seg in segments])

    return {"text": text}
