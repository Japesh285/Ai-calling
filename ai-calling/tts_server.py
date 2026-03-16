import asyncio
import io
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig


def _load_tts() -> TTS:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.serialization.safe_globals(
        [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
    ):
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


class SynthesizeRequest(BaseModel):
    text: str
    speaker: str = "Ana Florence"
    language: str = "en"


app = FastAPI(title="TTS Worker")
tts_model = _load_tts()
tts_model_lock = threading.Lock()


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "tts worker running"}


def _synthesize_wav_bytes(text: str, speaker: str, language: str) -> bytes:
    # XTTS inference is not thread-safe when sharing one in-process model.
    with tts_model_lock:
        wav = tts_model.tts(text=text, speaker=speaker, language=language)
        buffer = io.BytesIO()
        tts_model.synthesizer.save_wav(wav=wav, path=buffer)
        return buffer.getvalue()


@app.post("/synthesize")
async def synthesize(payload: SynthesizeRequest) -> Response:
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is required")

    wav_bytes = await asyncio.to_thread(
        _synthesize_wav_bytes,
        clean_text,
        payload.speaker,
        payload.language,
    )
    if not wav_bytes:
        raise HTTPException(status_code=500, detail="TTS generated no audio")

    return Response(content=wav_bytes, media_type="audio/wav")
