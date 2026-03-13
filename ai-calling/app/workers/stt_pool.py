from typing import Optional
import io
import wave

import httpx

from app.config.settings import PCM_CHANNELS, PCM_SAMPLE_RATE, PCM_SAMPLE_WIDTH_BYTES, STT_WORKERS
from app.utils.logger import get_logger

logger = get_logger(__name__)
_worker_index = 0


def _pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


async def transcribe(audio_bytes: bytes) -> Optional[str]:
    global _worker_index

    if not audio_bytes:
        return None

    worker = STT_WORKERS[_worker_index]
    _worker_index = (_worker_index + 1) % len(STT_WORKERS)

    wav_bytes = _pcm_to_wav_bytes(audio_bytes)
    files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}
    logger.info(
        "Calling STT worker %s with pcm_bytes=%s wav_bytes=%s",
        worker,
        len(audio_bytes),
        len(wav_bytes),
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(worker, files=files)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.exception("STT request failed for worker %s: %s", worker, exc)
        return None

    text = data.get("text") if isinstance(data, dict) else None
    if not text:
        logger.info("STT worker %s returned empty text", worker)
        return None

    logger.info("STT worker %s returned text", worker)
    return str(text).strip() or None
