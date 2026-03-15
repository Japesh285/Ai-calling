import io
import wave

from fastapi import FastAPI, HTTPException, UploadFile

from app.stt.faster_whisper_stt import FasterWhisperSTT


app = FastAPI()
stt = FasterWhisperSTT()


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise HTTPException(status_code=400, detail="Audio must be mono")
            if wav_file.getsampwidth() != 2:
                raise HTTPException(status_code=400, detail="Audio must be 16-bit PCM")

            pcm_bytes = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV upload: {exc}") from exc

    text = stt.transcribe(pcm_bytes)
    return {"text": text, "language": stt.last_detected_language}
