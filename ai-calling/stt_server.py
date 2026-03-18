import io
import wave

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

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


@app.post("/transcribe/stream")
async def transcribe_stream(audio: UploadFile):
    """
    Streaming transcription endpoint.
    Accepts raw PCM audio chunks and returns incremental transcriptions.
    """
    stt.init_stream()

    async def generate_transcripts():
        try:
            async for chunk in audio.stream():
                if not chunk:
                    continue

                transcript = stt.stream_transcribe(chunk)
                if transcript:
                    yield f"data: {transcript}\n\n"
        except Exception as exc:
            yield f"error: {exc}\n\n"
        finally:
            final_transcript = stt.finalize_stream()
            if final_transcript:
                yield f"data: {final_transcript}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_transcripts(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
