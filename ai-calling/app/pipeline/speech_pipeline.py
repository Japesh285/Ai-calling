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

# Deduplication and state tracking
_last_transcript = ""
_llm_in_progress = False
_last_speech_time = 0.0
_stt_request_time = 0.0


@dataclass(slots=True)
class SegmentationResult:
    segment_completed: bool = False
    segment_audio: bytes = b""
    streaming_stt_triggered: bool = False
    streaming_audio: bytes = b""
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
        self._streaming_sent = False
        self._utterance_id = 0
        self._stream_buffer = bytearray()

    def reset(self) -> None:
        self._buffer.clear()
        self._state = "silence"
        self._speech_ms = 0
        self._silence_frames = 0
        self._speech_started_at = None
        self._streaming_sent = False
        self._stream_buffer.clear()

    def process_frame(self, frame: bytes) -> SegmentationResult:
        rms = audioop.rms(frame, PCM_SAMPLE_WIDTH_BYTES) if frame else 0
        result = SegmentationResult(rms=rms)
        is_speech_start = rms > SPEECH_START_RMS_THRESHOLD
        is_speech_stop = rms < SPEECH_STOP_RMS_THRESHOLD

        if self._state == "silence":
            if is_speech_start:
                self._state = "speech"
                self._buffer.extend(frame)
                self._stream_buffer.extend(frame)
                self._speech_ms = FRAME_DURATION_MS
                self._silence_frames = 0
                self._speech_started_at = time.monotonic()
                self._streaming_sent = False
                self._utterance_id += 1
                result.speech_started = True
                result.utterance_id = self._utterance_id
                logger.info("Speech start detected")
            return result

        result.utterance_id = self._utterance_id
        self._buffer.extend(frame)
        self._stream_buffer.extend(frame)
        self._speech_ms += FRAME_DURATION_MS

        # Trigger streaming STT every ~500ms of speech for incremental updates
        if (
            self._speech_started_at is not None
            and not self._streaming_sent
            and time.monotonic() - self._speech_started_at >= 0.5
        ):
            result.streaming_stt_triggered = True
            result.streaming_audio = bytes(self._stream_buffer)
            self._streaming_sent = True
            self._stream_buffer.clear()
            logger.info("Streaming STT triggered")

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


async def transcribe_audio_streaming(audio_chunk: bytes) -> str:
    """
    Stream audio chunk to STT worker for incremental transcription.
    Uses the streaming endpoint for real-time updates.
    """
    if not audio_chunk:
        return ""

    worker = STT_WORKERS[0]
    wav_bytes = _pcm_to_wav_bytes(audio_chunk)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{worker}/stream",
                content=io.BytesIO(wav_bytes),
                headers={"Content-Type": "audio/wav"},
            ) as response:
                response.raise_for_status()
                latest_transcript = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        if data.startswith("error: "):
                            logger.error("Streaming STT error: %s", data[7:])
                            continue
                        latest_transcript = data

                return latest_transcript
    except httpx.HTTPError as exc:
        logger.exception("Streaming STT request failed: %s", exc)
        return ""


async def transcribe_audio(audio_bytes: bytes, is_final_transcript: bool = True) -> str:
    global _last_transcript, _stt_request_time
    if not audio_bytes:
        return ""

    # Only process final transcripts for LLM triggering
    if not is_final_transcript:
        return ""

    _stt_request_time = time.monotonic()
    logger.info("STT request sent at %.3f", _stt_request_time)

    worker = STT_WORKERS[0]
    wav_bytes = _pcm_to_wav_bytes(audio_bytes)
    files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(worker, files=files)
        response.raise_for_status()
        data = response.json()

    text = data.get("text") if isinstance(data, dict) else ""
    transcript = str(text or "").strip().lower()

    # Ignore system/garbage text
    if "the following conversation may contain" in transcript:
        return ""

    # Ignore invalid or too-short transcripts
    if len(transcript) < 4 or transcript in {"", ".", "i", "uh", "um"}:
        return ""

    # Deduplicate transcripts
    if transcript == _last_transcript:
        return ""
    _last_transcript = transcript

    elapsed_ms = (time.monotonic() - _stt_request_time) * 1000
    logger.info("STT transcript received in %.0fms: %s", elapsed_ms, transcript)
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
    global _last_transcript, _llm_in_progress, _last_speech_time
    speech_stop_time = time.monotonic()
    logger.info("Speech stop detected at %.3f", speech_stop_time)
    
    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        return "", ""

    # Prevent multiple LLM calls
    if _llm_in_progress:
        logger.info("LLM already in progress, skipping duplicate")
        return "", ""

    # Reduced cooldown check (0.5s → 0.1s for faster response)
    now = time.monotonic()
    if now - _last_speech_time < 0.1:
        logger.info("Speech segment too soon, skipping")
        return "", ""
    _last_speech_time = now

    _llm_in_progress = True
    try:
        response = await ask_brain(transcript)
        await _append_brain_response(transcript, response)
        return transcript, response
    finally:
        _llm_in_progress = False


async def process_speech_segment_to_reply_streaming(
    audio_bytes: bytes,
    sentence_queue: asyncio.Queue[str | None],
) -> tuple[str, str]:
    global _last_transcript, _llm_in_progress, _last_speech_time
    speech_stop_time = time.monotonic()
    logger.info("Speech stop detected at %.3f", speech_stop_time)
    
    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        return "", ""

    # Prevent multiple LLM calls
    if _llm_in_progress:
        logger.info("LLM already in progress, skipping duplicate")
        return "", ""

    # Reduced cooldown check (0.5s → 0.1s for faster response)
    now = time.monotonic()
    if now - _last_speech_time < 0.1:
        logger.info("Speech segment too soon, skipping")
        return "", ""
    _last_speech_time = now

    _llm_in_progress = True
    try:
        response = await process_transcript_to_reply_streaming(transcript, sentence_queue)
        await _append_brain_response(transcript, response)
        return transcript, response
    finally:
        _llm_in_progress = False


async def process_transcript_to_reply_streaming(
    transcript: str,
    sentence_queue: asyncio.Queue[str | None],
) -> str:
    global _llm_in_progress, _last_speech_time
    if not transcript:
        return ""

    # Prevent multiple LLM calls
    if _llm_in_progress:
        logger.info("LLM already in progress, skipping duplicate")
        return ""

    # Segment cooldown check
    now = time.monotonic()
    if now - _last_speech_time < 0.5:
        logger.info("Speech segment too soon, skipping")
        return ""
    _last_speech_time = now

    _llm_in_progress = True
    try:
        logger.info("LLM streaming started")
        response_buffer = ""
        chunk_buffer = ""
        chunk_start_time = time.monotonic()
        MIN_CHUNK_CHARS = 50
        MAX_CHUNK_MS = 300

        def _is_safe_chunk_boundary(text: str) -> bool:
            """Check if text ends at a safe boundary (space or punctuation)."""
            if not text:
                return False
            last_char = text[-1]
            return last_char.isspace() or last_char in {".", "!", "?", ",", ";", ":", ")", "]", "\"", "'"}

        async for token in stream_brain_response(transcript):
            response_buffer += token
            chunk_buffer += token

            elapsed_ms = (time.monotonic() - chunk_start_time) * 1000
            should_flush = (
                len(chunk_buffer) >= MIN_CHUNK_CHARS
                or (len(chunk_buffer) >= 20 and elapsed_ms >= MAX_CHUNK_MS)
            )

            if should_flush and chunk_buffer.strip():
                # Find last safe boundary to avoid breaking mid-word
                flush_text = chunk_buffer
                if not _is_safe_chunk_boundary(chunk_buffer):
                    # Look backwards for last safe boundary
                    for i in range(len(chunk_buffer) - 1, max(0, len(chunk_buffer) - 20), -1):
                        if _is_safe_chunk_boundary(chunk_buffer[i]):
                            flush_text = chunk_buffer[:i + 1]
                            break

                if flush_text.strip():
                    logger.info("LLM chunk ready for TTS")
                    await sentence_queue.put(flush_text.strip())
                    chunk_buffer = chunk_buffer[len(flush_text):]
                    chunk_start_time = time.monotonic()

        if chunk_buffer.strip():
            logger.info("LLM chunk ready for TTS")
            await sentence_queue.put(chunk_buffer.strip())

        response = response_buffer.strip()
        logger.info("AI response received: %s", response)
        return response
    finally:
        _llm_in_progress = False


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
