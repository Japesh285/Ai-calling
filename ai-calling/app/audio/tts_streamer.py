from __future__ import annotations

import asyncio
import audioop
import base64
import json

from fastapi import WebSocket

from app.config.settings import FRAME_DURATION_MS, PCM_SAMPLE_RATE, PCMA_FRAME_BYTES
from app.utils.logger import get_logger
from app.workers.tts_pool import synthesize

logger = get_logger(__name__)


async def generate_tts(text: str, language: str | None = None) -> bytes:
    pcm16 = await synthesize(text, language)
    if not pcm16:
        return b""
    logger.info("TTS generated")
    return pcm16


def _convert_pcm16_to_pcma(pcm16_bytes: bytes) -> bytes:
    """Convert PCM16 (16kHz) to PCMA (8kHz) using audioop - no ffmpeg, no disk I/O."""
    if not pcm16_bytes:
        return b""

    # Downsample from 16kHz to 8kHz
    pcm16_8k, _ = audioop.ratecv(pcm16_bytes, 2, 1, PCM_SAMPLE_RATE, 8000, None)
    # Convert PCM16 to A-law
    pcma_bytes = audioop.lin2alaw(pcm16_8k, 2)
    logger.info("TTS converted to PCMA (in-memory)")
    logger.info("PCMA size: %s", len(pcma_bytes))
    logger.info("Frames to stream: %s", len(pcma_bytes) // PCMA_FRAME_BYTES)
    return pcma_bytes


async def convert_pcm16_bytes_to_pcma(pcm16_bytes: bytes) -> bytes:
    """In-memory PCM16 to PCMA conversion."""
    return _convert_pcm16_to_pcma(pcm16_bytes)


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
