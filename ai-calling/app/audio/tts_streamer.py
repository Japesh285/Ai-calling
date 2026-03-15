from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import tempfile
import wave

from fastapi import WebSocket

from app.config.settings import FRAME_DURATION_MS, PCM_CHANNELS, PCM_SAMPLE_RATE, PCM_SAMPLE_WIDTH_BYTES, PCMA_FRAME_BYTES
from app.utils.logger import get_logger
from app.workers.tts_pool import synthesize

logger = get_logger(__name__)


async def generate_tts(text: str) -> bytes:
    pcm16 = await synthesize(text)
    if not pcm16:
        return b""
    logger.info("TTS generated")
    return pcm16


def _write_pcm16_wav(pcm16_bytes: bytes, output_path: Path) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(pcm16_bytes)


async def convert_to_pcma(pcm16_wav_path: str) -> bytes:
    if not pcm16_wav_path:
        return b""

    with tempfile.NamedTemporaryFile(suffix=".alaw", delete=False) as output_file:
        output_path = Path(output_file.name)

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        pcm16_wav_path,
        "-ar",
        "8000",
        "-ac",
        "1",
        "-acodec",
        "pcm_alaw",
        "-f",
        "alaw",
        str(output_path),
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(
            "ffmpeg PCMA conversion failed with code %s: %s",
            process.returncode,
            (stderr or b"").decode(errors="replace").strip(),
        )
        return b""

    file_process = await asyncio.create_subprocess_exec(
        "file",
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    file_stdout, _ = await file_process.communicate()
    if file_stdout:
        logger.info("file %s", file_stdout.decode(errors="replace").strip())

    pcma_bytes = output_path.read_bytes()
    output_path.unlink(missing_ok=True)
    logger.info("TTS converted to PCMA")
    logger.info("PCMA size: %s", len(pcma_bytes))
    logger.info("Frames to stream: %s", len(pcma_bytes) // PCMA_FRAME_BYTES)
    return pcma_bytes


async def convert_pcm16_bytes_to_pcma(pcm16_bytes: bytes) -> bytes:
    if not pcm16_bytes:
        return b""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = Path(wav_file.name)

    _write_pcm16_wav(pcm16_bytes, wav_path)
    try:
        return await convert_to_pcma(str(wav_path))
    finally:
        wav_path.unlink(missing_ok=True)


def _build_stream_audio_message(frame: bytes) -> dict[str, object]:
    return {
        "type": "streamAudio",
        "data": {
            "audioDataType": "pcma",
            "sampleRate": 8000,
            "audioData": base64.b64encode(frame).decode("ascii"),
        },
    }


async def stream_pcma_audio(websocket: WebSocket, pcma_bytes: bytes) -> None:
    if not pcma_bytes:
        return

    frame_duration_seconds = FRAME_DURATION_MS / 1000.0
    frame_count = len(pcma_bytes) // PCMA_FRAME_BYTES
    logger.info("Streaming PCMA frames: %d", frame_count)

    for frame_index, start in enumerate(range(0, len(pcma_bytes), PCMA_FRAME_BYTES), start=1):
        frame = pcma_bytes[start : start + PCMA_FRAME_BYTES]
        if len(frame) != PCMA_FRAME_BYTES:
            break

        message = _build_stream_audio_message(frame)
        if frame_index <= 10:
            logger.info("Streaming frame %s", frame_index)
            logger.info("Sending payload: %s", json.dumps(message)[:160])
        await websocket.send_json(message)
        await asyncio.sleep(frame_duration_seconds)


async def test_noise(websocket: WebSocket) -> None:
    for _ in range(200):
        await websocket.send_json(_build_stream_audio_message(b"\xff" * PCMA_FRAME_BYTES))
        await asyncio.sleep(0.02)


async def debug_play_pcma(pcma_path: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffplay",
        "-autoexit",
        "-nodisp",
        "-f",
        "alaw",
        "-ar",
        "8000",
        "-ac",
        "1",
        pcma_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.communicate()


async def stream_tts(websocket: WebSocket, pcma_audio: bytes) -> None:
    await stream_pcma_audio(websocket, pcma_audio)


async def stream_pcma(websocket: WebSocket, pcma_bytes: bytes) -> None:
    await stream_pcma_audio(websocket, pcma_bytes)
