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
    STT_LOG_RAW_ON_FAILURE,
    STT_WORKERS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_IGNORED_TRANSCRIPTS = {
    "",
    ".",
    "i",
    "uh",
    "um",
    "hello",
    "hello.",
    "hello, hello.",
    "thanks for watching!",
    "thanks for watching.",
    "you",
}

# Fixed chunk size configuration for uniform TTS generation
TARGET_CHUNK_WORDS = 5  # Target chunk size
MIN_CHUNK_WORDS = 5     # Minimum chunk size (except last)
MAX_CHUNK_WORDS = 6     # Maximum chunk size for even distribution
SINGLE_CHUNK_THRESHOLD = 7  # Sentences ≤7 words stay as one chunk

# Hindi sentence boundary markers
HINDI_PUNCTUATION = {"।", "॥", "?", "!", ".", ",", ";"}

# Pipeline scheduling configuration
# Delay between putting consecutive text chunks on the TTS sentence queue.
#
# Set to 0: all chunks are queued back-to-back. Safe because the gateway uses a
# single TTS worker that processes sentence_queue strictly in order — chunk N+1
# cannot start until chunk N's streaming TTS + playback completes. The old 2.5s
# stagger only added dead air between phrases (caller heard long silences).
#
# If you add parallel TTS workers later, you may need a small delay again so
# playback order matches chunk order — tune empirically.
PIPELINE_DELAY_MS = 0
MIN_PIPELINE_CHUNKS = 2   # Reserved for future multi-worker pipelining

# When final STT returns empty, retry if audio looks like real speech (reduces flaky blanks).
STT_EMPTY_RETRY_EXTRA = 2
STT_EMPTY_RETRY_DELAY_SEC = 0.25
STT_EMPTY_RETRY_MIN_AUDIO_MS = 600
STT_EMPTY_RETRY_MIN_RMS = 100

# Adaptive delay tracking (tracks actual TTS compute times)
_tts_compute_times: list[float] = []

# Deduplication and state tracking
_last_transcript = ""
_llm_in_progress = False
_last_speech_time = 0.0
_stt_request_time = 0.0


def split_into_fixed_chunks(text: str) -> list[str]:
    return shape_chunks(text)


def shape_chunks(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    min_words = 4
    max_words = 7
    weak_starts = {"है", "हैं", "हूँ"}
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    sentence_chunks = _split_at_punctuation(text, words) or [text.strip()]

    merged_chunks: list[str] = []
    index = 0
    while index < len(sentence_chunks):
        current = sentence_chunks[index].strip()
        while len(current.split()) < min_words and index + 1 < len(sentence_chunks):
            index += 1
            current = f"{current} {sentence_chunks[index].strip()}".strip()
        merged_chunks.append(current)
        index += 1

    balanced_chunks: list[str] = []
    for chunk in merged_chunks:
        chunk_words = chunk.split()
        if len(chunk_words) <= max_words:
            balanced_chunks.append(chunk)
            continue

        remaining = chunk_words[:]
        while remaining:
            remaining_count = len(remaining)
            if remaining_count <= max_words:
                balanced_chunks.append(" ".join(remaining))
                break

            split_size = 6 if remaining_count - 6 == 0 or remaining_count - 6 >= min_words else 5
            split_size = min(split_size, remaining_count)
            split_size = _adjust_for_auxiliary(remaining, split_size)

            if remaining_count - split_size < min_words and remaining_count > max_words:
                split_size = max(min_words, remaining_count - min_words)
                split_size = _adjust_for_auxiliary(remaining, split_size)

            balanced_chunks.append(" ".join(remaining[:split_size]))
            remaining = remaining[split_size:]

    final_chunks: list[str] = []
    for chunk in balanced_chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        first_word = stripped.split()[0]
        if final_chunks and (len(stripped.split()) < min_words or first_word in weak_starts):
            final_chunks[-1] = f"{final_chunks[-1]} {stripped}".strip()
        else:
            final_chunks.append(stripped)

    if len(final_chunks) == 1:
        return final_chunks

    logger.info(
        "Shaped chunks: %s sizes=%s text=%s",
        len(final_chunks),
        [len(chunk.split()) for chunk in final_chunks],
        text[:80],
    )
    return final_chunks


def _split_at_punctuation(text: str, words: list[str]) -> list[str]:
    """
    Split text at Hindi punctuation marks (। ? ! ,).
    Returns list of sentences if multiple found, else empty list.
    """
    sentences = []
    current_start = 0
    
    for i, word in enumerate(words):
        # Check if word ends with punctuation
        if any(word.endswith(p) for p in HINDI_PUNCTUATION):
            sentence = " ".join(words[current_start:i + 1])
            if sentence.strip():
                sentences.append(sentence)
            current_start = i + 1
    
    # Only return if we found 2+ sentences
    if len(sentences) >= 2:
        # Add remaining text as last sentence
        if current_start < len(words):
            remaining = " ".join(words[current_start:])
            if remaining.strip():
                sentences.append(remaining)
        return sentences
    
    return []


def _adjust_for_auxiliary(words: list[str], proposed_split: int) -> int:
    """
    Adjust split point to avoid breaking before auxiliary verbs.
    Hindi auxiliaries: हैं, है, हूँ, था, थी, थे, होगा, होगी
    """
    auxiliaries = {"हैं", "है", "हूँ", "था", "थी", "थे", "होगा", "होगी", "हो", "रहा", "रही", "रहे"}
    
    # If the word AFTER split is an auxiliary, include it in first chunk
    if proposed_split < len(words) and words[proposed_split] in auxiliaries:
        return min(proposed_split + 1, len(words) - 1)
    
    # If the word BEFORE split is an auxiliary, keep it in first chunk
    if proposed_split > 0 and words[proposed_split - 1] in auxiliaries:
        return proposed_split
    
    return proposed_split


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


def _pcm_duration_ms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    bytes_per_sec = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
    return (len(pcm) / bytes_per_sec) * 1000.0


def _audio_substantial_for_stt_retry(pcm: bytes) -> bool:
    """True if segment is long/loud enough that an empty STT is likely transient."""
    if _pcm_duration_ms(pcm) < STT_EMPTY_RETRY_MIN_AUDIO_MS:
        return False
    rms = audioop.rms(pcm, PCM_SAMPLE_WIDTH_BYTES)
    return rms >= STT_EMPTY_RETRY_MIN_RMS


def _looks_like_garbled_hindi(transcript: str, stt_meta: Optional[dict] = None) -> bool:
    """
    Heuristic flags for logging — wrong answers often follow bad STT.
    Whisper avg_logprob typically < -0.8 when the model is uncertain.
    """
    if stt_meta:
        lp = stt_meta.get("avg_logprob")
        if lp is not None:
            try:
                if float(lp) < -0.78:
                    return True
            except (TypeError, ValueError):
                pass
    t = transcript.lower()
    suspicious = (
        "वानकारी",
        "वानकार",
        "thank you for watching",
        "thanks for watching",
    )
    if any(s in t for s in suspicious):
        return True
    # Mostly Latin on a Hindi-first line — likely mis-detected language
    letters = [c for c in transcript if c.isalpha()]
    if letters:
        latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
        if latin / len(letters) > 0.65 and len(transcript) > 8:
            return True
    return False


def _filter_stt_transcript(transcript: str) -> str:
    """Return usable text or empty if we should treat as no transcript."""
    if not transcript:
        return ""
    if (
        "the following conversation may contain" in transcript
        or "this conversation is with an indian tax support assistant" in transcript
    ):
        return ""
    if len(transcript) < 4 or transcript in _IGNORED_TRANSCRIPTS:
        return ""
    return transcript


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

    max_attempts = 1 + STT_EMPTY_RETRY_EXTRA
    substantial = _audio_substantial_for_stt_retry(audio_bytes)

    async with httpx.AsyncClient(timeout=45.0) as client:
        for attempt in range(max_attempts):
            if attempt > 0:
                await asyncio.sleep(STT_EMPTY_RETRY_DELAY_SEC)
                dur = _pcm_duration_ms(audio_bytes)
                rms = audioop.rms(audio_bytes, PCM_SAMPLE_WIDTH_BYTES)
                logger.info(
                    "STT retry %s/%s (duration=%.0fms rms=%s)",
                    attempt + 1,
                    max_attempts,
                    dur,
                    rms,
                )

            response = await client.post(worker, files=files)
            response.raise_for_status()
            data = response.json()
            text = data.get("text") if isinstance(data, dict) else ""
            raw = str(text or "").strip().lower()
            candidate = _filter_stt_transcript(raw)

            if STT_LOG_RAW_ON_FAILURE and isinstance(data, dict):
                meta = {k: v for k, v in data.items() if k != "text"}
                if not candidate:
                    logger.warning(
                        "STT unusable/filtered transcript — full 8001 JSON: %s",
                        data,
                    )
                elif _looks_like_garbled_hindi(candidate, data):
                    logger.warning(
                        "STT possible garble — text=%r meta=%s",
                        candidate,
                        meta,
                    )

            if candidate:
                if candidate == _last_transcript:
                    return ""
                _last_transcript = candidate
                elapsed_ms = (time.monotonic() - _stt_request_time) * 1000
                logger.info("STT transcript received in %.0fms: %s", elapsed_ms, candidate)
                return candidate

            # Empty or filtered out: retry only if audio looks like real speech
            if attempt < max_attempts - 1 and substantial:
                continue
            break

    if substantial:
        logger.warning(
            "STT empty after %s attempt(s); duration=%.0fms rms=%s",
            max_attempts,
            _pcm_duration_ms(audio_bytes),
            audioop.rms(audio_bytes, PCM_SAMPLE_WIDTH_BYTES),
        )
    return ""


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
        llm_start_time = time.monotonic()

        # Accumulate full LLM response first
        async for token in stream_brain_response(transcript):
            response_buffer += token

        llm_duration = (time.monotonic() - llm_start_time) * 1000
        response = response_buffer.strip()
        logger.info("AI response received in %.0fms: %s", llm_duration, response)

        # One TTS job per assistant reply when under a safe size (fewer round-trips,
        # pairs with single uuid_broadcast per turn in websocket non-FIFO mode).
        _max_single = 8000
        if len(response) <= _max_single:
            fixed_chunks = [response] if response else []
        else:
            fixed_chunks = split_into_fixed_chunks(response)

        if not fixed_chunks:
            logger.info("No chunks generated from response")
            return response

        # Schedule chunks through pipeline dispatcher
        await schedule_tts_pipeline(fixed_chunks, sentence_queue)

        return response
    finally:
        _llm_in_progress = False


async def schedule_tts_pipeline(
    chunks: list[str],
    sentence_queue: asyncio.Queue[str | None],
) -> None:
    """
    Enqueue text chunks for the TTS worker(s).

    With PIPELINE_DELAY_MS == 0, all chunks are queued immediately; the single
    worker drains the queue in order, so there is no need for artificial sleep
    between puts.
    """
    total_chunks = len(chunks)
    pipeline_delay_sec = PIPELINE_DELAY_MS / 1000.0

    logger.info(
        "Pipeline scheduler: %s chunks, inter-dispatch delay=%.2fs",
        total_chunks,
        pipeline_delay_sec,
    )

    for chunk_index, chunk in enumerate(chunks):
        dispatch_time = time.monotonic()
        word_count = len(chunk.split())

        logger.info(
            "Pipeline dispatch: chunk %s/%s (%s words) at t+%.2fs: %s",
            chunk_index + 1,
            total_chunks,
            word_count,
            dispatch_time % 1000,
            chunk[:50],
        )

        await sentence_queue.put(chunk)

        if chunk_index < total_chunks - 1 and pipeline_delay_sec > 0:
            await asyncio.sleep(pipeline_delay_sec)
            logger.info(
                "Pipeline delay: %.2fs elapsed, ready for chunk %s",
                pipeline_delay_sec,
                chunk_index + 2,
            )

    logger.info("Pipeline scheduler: all %s chunks dispatched", total_chunks)


async def append_brain_response(transcript: str, response: str) -> None:
    await _append_brain_response(transcript, response)


def _extract_completed_sentences(buffer: str) -> tuple[list[str], str]:
    completed_sentences: list[str] = []
    sentence_start = 0

    for index, char in enumerate(buffer):
        if char in {".", "!", "?", "।", "॥", "?"}:  # includes Devanagari danda
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
