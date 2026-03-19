import asyncio
import audioop
import io
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Optional
import wave

import httpx
import numpy as np
from scipy.signal import resample_poly

from app.config.settings import (
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    TTS_AUDIO_DUMP_DIR,
    TTS_WORKERS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
_tts_worker_index = 0


_ROMANIZED_HINDI_WORDS = frozenset({
    # Greetings — unambiguously Hindi
    "namaste", "namaskar", "shukriya", "dhanyawad",
    # Pronouns — unambiguously Hindi (avoid "main"/"tum" which overlap English)
    "aap", "hum", "woh", "yeh", "mujhe", "aapko", "unhe",
    # Auxiliaries — unambiguous Hindi forms (avoid "the"/"be" which are English)
    "hai", "hain", "tha", "thi", "hoga", "hogi", "hote", "hoti",
    # Verbs — distinctly Hindi
    "karo", "karna", "karein", "karte", "karti", "kijiye",
    "batao", "batayein", "samjhiye", "chahiye", "chahte",
    "milega", "milegi", "rahega", "rahegi",
    # Question words — all unambiguously Hindi
    "kya", "kaise", "kab", "kahan", "kyun", "kaun", "kitna", "kitne",
    # Particles — distinctly Hindi
    "ka", "ki", "ke", "ko", "mein", "se", "aur", "lekin",
    "toh", "bhi", "nahi", "zaroor", "bilkul",
    # Nouns — distinctly Hindi
    "madad", "zaroorat", "kaam", "paisa", "rupaye",
    "sawaal", "jawab", "jankari", "jaankari", "baat", "cheez",
})


def _detect_language(text: str) -> str:
    """
    Detect Hindi content in either Devanagari script or Romanized Latin form.
    XTTS voice cloning works correctly only when the language matches the text.
    The brain often replies in Romanized Hindi (e.g. 'Aap kaise hain?') which
    must be synthesised with language='hi' — not 'en' — for natural pronunciation.
    Threshold is 2 unambiguous Hindi words to avoid English false positives.
    """
    # Devanagari script — definitive
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return "hi"

    # Romanized Hindi — count unambiguous Hindi word hits
    words = text.lower().split()
    hindi_hits = sum(1 for w in words if w.strip(".,!?\"'();") in _ROMANIZED_HINDI_WORDS)
    if hindi_hits >= 2:
        return "hi"

    return "en"


_TTS_DUMP_MAX_FILES = 20


def _dump_tts_audio(audio_bytes: bytes) -> Optional[Path]:
    if not audio_bytes:
        return None

    TTS_AUDIO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = TTS_AUDIO_DUMP_DIR / f"tts_{timestamp}.wav"

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)

    # Rolling retention: delete oldest files beyond the cap
    existing = sorted(TTS_AUDIO_DUMP_DIR.glob("tts_*.wav"))
    for old_file in existing[:-_TTS_DUMP_MAX_FILES]:
        try:
            old_file.unlink()
        except Exception:
            pass

    return output_path


def _normalize_tts_audio(audio_bytes: bytes) -> Optional[bytes]:
    if not audio_bytes:
        return None

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error:
        logger.warning("TTS worker did not return WAV audio")
        return audio_bytes

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1

    if sample_width != PCM_SAMPLE_WIDTH_BYTES:
        frames = audioop.lin2lin(frames, sample_width, PCM_SAMPLE_WIDTH_BYTES)
        sample_width = PCM_SAMPLE_WIDTH_BYTES

    if sample_rate != PCM_SAMPLE_RATE:
        # scipy.signal.resample_poly applies a proper anti-aliasing FIR filter
        # before decimation, preventing aliasing distortion from XTTS 24kHz → 8kHz
        pcm_array = np.frombuffer(frames, dtype=np.int16)
        g = gcd(PCM_SAMPLE_RATE, sample_rate)
        up = PCM_SAMPLE_RATE // g
        down = sample_rate // g
        resampled = resample_poly(pcm_array, up, down)
        frames = np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    return frames


async def synthesize(text: str, language: Optional[str] = None) -> Optional[bytes]:
    global _tts_worker_index
    clean_text = (text or "").strip()
    if not clean_text:
        return None

    detected_language = language if language else _detect_language(clean_text)
    logger.info("Starting TTS worker for text length=%s language=%s", len(clean_text), detected_language)
    worker = TTS_WORKERS[_tts_worker_index % len(TTS_WORKERS)]
    _tts_worker_index += 1
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            worker,
            json={"text": clean_text, "language": detected_language}
        )
        if response.status_code != 200:
            logger.error(
                "TTS worker returned HTTP %s for language=%s text=%r: %s",
                response.status_code,
                detected_language,
                clean_text[:80],
                response.text[:200],
            )
            return None
        raw_audio = response.content

    normalized_audio = _normalize_tts_audio(raw_audio)
    if normalized_audio:
        logger.info("TTS produced outbound audio bytes=%s", len(normalized_audio))
        dump_path = await asyncio.to_thread(_dump_tts_audio, normalized_audio)
        if dump_path:
            logger.info("Saved TTS audio dump to %s", dump_path)
    else:
        logger.info("TTS produced no outbound audio")
    return normalized_audio
