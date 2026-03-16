from __future__ import annotations

import asyncio
import audioop
import io
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.config.settings import (
    AI_BRAIN_VOICE_URL,
    AI_BRAIN_RESPONSE_PATH,
    FRAME_DURATION_MS,
    MAX_SPEECH_LENGTH_MS,
    MIN_SPEECH_LENGTH_MS,
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    SILENCE_STOP_FRAMES,
    SILENCE_TIMEOUT_MS,
    SPEECH_STOP_RMS_THRESHOLD,
    SPEECH_START_RMS_THRESHOLD,
    STT_WORKERS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SegmentationResult:
    segment_completed: bool = False
    segment_audio: bytes = b""
    partial_stt_triggered: bool = False
    partial_audio: bytes = b""
    speech_started: bool = False
    speech_stopped: bool = False
    utterance_id: int = 0
    rms: int = 0


class SpeechSegmenter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._state = "silence"
        self._speech_ms = 0
        self._silence_frames = 0
        self._speech_started_at: float | None = None
        self._partial_stt_sent = False
        self._utterance_id = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._state = "silence"
        self._speech_ms = 0
        self._silence_frames = 0
        self._speech_started_at = None
        self._partial_stt_sent = False

    def process_frame(self, frame: bytes) -> SegmentationResult:
        rms = audioop.rms(frame, PCM_SAMPLE_WIDTH_BYTES) if frame else 0
        result = SegmentationResult(rms=rms)
        is_speech_start = rms > SPEECH_START_RMS_THRESHOLD
        is_speech_stop = rms < SPEECH_STOP_RMS_THRESHOLD

        if self._state == "silence":
            if is_speech_start:
                self._state = "speech"
                self._buffer.extend(frame)
                self._speech_ms = FRAME_DURATION_MS
                self._silence_frames = 0
                self._speech_started_at = time.monotonic()
                self._partial_stt_sent = False
                self._utterance_id += 1
                result.speech_started = True
                result.utterance_id = self._utterance_id
                logger.info("Speech start detected")
            return result

        result.utterance_id = self._utterance_id
        self._buffer.extend(frame)
        self._speech_ms += FRAME_DURATION_MS

        if (
            self._speech_started_at is not None
            and not self._partial_stt_sent
            and time.monotonic() - self._speech_started_at > 1.0
        ):
            result.partial_stt_triggered = True
            result.partial_audio = bytes(self._buffer)
            self._partial_stt_sent = True
            logger.info("Partial STT triggered")

        if is_speech_stop:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        if (
            self._silence_frames >= SILENCE_STOP_FRAMES
            or self._speech_ms >= MAX_SPEECH_LENGTH_MS
        ):
            segment_duration_ms = self._speech_ms - (self._silence_frames * FRAME_DURATION_MS)
            result.speech_stopped = True
            result.segment_completed = True
            result.segment_audio = self._finalize_segment(trim_silence_frames=self._silence_frames)
            if segment_duration_ms < MIN_SPEECH_LENGTH_MS:
                logger.info("Silence chunk ignored")
                result.segment_audio = b""
            else:
                logger.info("Final STT triggered")
                logger.info("Speech stop detected")
        return result

    def flush(self) -> SegmentationResult:
        result = SegmentationResult(utterance_id=self._utterance_id)
        if self._state != "speech":
            return result

        segment_duration_ms = self._speech_ms - (self._silence_frames * FRAME_DURATION_MS)
        result.segment_audio = self._finalize_segment(trim_silence_frames=self._silence_frames)
        if segment_duration_ms < MIN_SPEECH_LENGTH_MS:
            logger.info("Silence chunk ignored")
            result.segment_audio = b""
            return result
        result.segment_completed = True
        result.speech_stopped = True
        logger.info("Final STT triggered")
        logger.info("Speech stop detected")
        return result

    def _finalize_segment(self, trim_silence_frames: int = 0) -> bytes:
        trim_bytes = trim_silence_frames * FRAME_DURATION_MS * PCM_SAMPLE_WIDTH_BYTES * PCM_CHANNELS * PCM_SAMPLE_RATE // 1000
        if trim_bytes > 0:
            segment_audio = bytes(self._buffer[:-trim_bytes]) if trim_bytes < len(self._buffer) else b""
        else:
            segment_audio = bytes(self._buffer)
        self.reset()
        return segment_audio


def _pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


async def transcribe_audio(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""

    worker = STT_WORKERS[0]
    wav_bytes = _pcm_to_wav_bytes(audio_bytes)
    files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(worker, files=files)
        response.raise_for_status()
        data = response.json()

    text = data.get("text") if isinstance(data, dict) else ""
    transcript = str(text or "").strip()
    if transcript:
        logger.info("Speech chunk sent to STT")
    logger.info("STT transcript: %s", transcript)
    return transcript


async def ask_brain(text: str) -> str:
    if not text:
        return ""

    chunks: list[str] = []
    async for chunk in stream_brain_response(text):
        chunks.append(chunk)

    brain_response = "".join(chunks).strip()
    logger.info("AI response received: %s", brain_response)
    return brain_response


async def stream_brain_response(text: str):
    if not text:
        return

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", AI_BRAIN_VOICE_URL, json={"query": text}) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                if chunk:
                    yield chunk


async def process_speech_segment(audio_bytes: bytes) -> None:
    try:
        transcript, response = await process_speech_segment_to_reply(audio_bytes)
        if not transcript:
            logger.info("STT returned empty transcript")
            return
        print(f"User: {transcript}")
        print(f"AI: {response}")
    except Exception:
        logger.exception("Speech segment processing failed")


async def process_speech_segment_to_reply(audio_bytes: bytes) -> tuple[str, str]:
    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        return "", ""

    response = await ask_brain(transcript)
    await _append_brain_response(transcript, response)
    return transcript, response


async def process_speech_segment_to_reply_streaming(
    audio_bytes: bytes,
    sentence_queue: asyncio.Queue[str | None],
) -> tuple[str, str]:
    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        return "", ""

    response = await process_transcript_to_reply_streaming(transcript, sentence_queue)
    await _append_brain_response(transcript, response)
    return transcript, response


async def process_transcript_to_reply_streaming(
    transcript: str,
    sentence_queue: asyncio.Queue[str | None],
) -> str:
    if not transcript:
        return ""

    logger.info("LLM streaming started")
    response_buffer = ""
    sentence_buffer = ""

    async for token in stream_brain_response(transcript):
        response_buffer += token
        sentence_buffer += token

        completed_sentences, sentence_buffer = _extract_completed_sentences(sentence_buffer)
        for sentence in completed_sentences:
            logger.info("LLM chunk ready for TTS")
            await sentence_queue.put(sentence)

    if sentence_buffer.strip():
        logger.info("LLM chunk ready for TTS")
        await sentence_queue.put(sentence_buffer.strip())

    response = response_buffer.strip()
    logger.info("AI response received: %s", response)
    return response


async def append_brain_response(transcript: str, response: str) -> None:
    await _append_brain_response(transcript, response)


def _extract_completed_sentences(buffer: str) -> tuple[list[str], str]:
    completed_sentences: list[str] = []
    sentence_start = 0

    for index, char in enumerate(buffer):
        if char in {".", "!", "?"}:
            sentence = buffer[sentence_start : index + 1].strip()
            if sentence:
                completed_sentences.append(sentence)
            sentence_start = index + 1

    return completed_sentences, buffer[sentence_start:]


async def _append_brain_response(transcript: str, response: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] User: {transcript}\n[{timestamp}] AI: {response}\n\n"

    def _write() -> None:
        AI_BRAIN_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AI_BRAIN_RESPONSE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)

    await asyncio.to_thread(_write)
