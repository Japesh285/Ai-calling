import asyncio
import audioop
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import wave

from app.config.settings import (
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    TTS_AUDIO_DUMP_DIR,
    TTS_COMMAND,
    TTS_SCRIPT_CWD,
    TTS_SCRIPT_NAME,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            sample_rate,
            PCM_SAMPLE_RATE,
            None,
        )

    return frames


async def synthesize(text: str) -> Optional[bytes]:
    clean_text = (text or "").strip()
    if not clean_text:
        return None

    script_path = Path(TTS_SCRIPT_CWD) / TTS_SCRIPT_NAME
    if not script_path.exists():
        logger.error("TTS script not found at %s", script_path)
        return None

    logger.info("Starting TTS worker for text length=%s", len(clean_text))
    process = await asyncio.create_subprocess_exec(
        *TTS_COMMAND,
        clean_text,
        cwd=str(TTS_SCRIPT_CWD),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(
            "TTS process failed with code %s: %s",
            process.returncode,
            (stderr or b"").decode(errors="replace").strip(),
        )
        return None

    if stderr:
        logger.info("TTS stderr: %s", stderr.decode(errors="replace").strip())

    normalized_audio = _normalize_tts_audio(stdout or b"")
    if normalized_audio:
        logger.info("TTS produced outbound audio bytes=%s", len(normalized_audio))
        dump_path = await asyncio.to_thread(_dump_tts_audio, normalized_audio)
        if dump_path:
            logger.info("Saved TTS audio dump to %s", dump_path)
    else:
        logger.info("TTS produced no outbound audio")
    return normalized_audio
