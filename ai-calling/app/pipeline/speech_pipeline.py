import asyncio
from datetime import datetime, timezone
from typing import Optional

from app.config.settings import STT_FLUSH_EVERY_N_CHUNKS, STT_FLUSH_INTERVAL_SECONDS, TRANSCRIPT_PATH
from app.sessions.call_session import CallSession
from app.utils.logger import get_logger
from app.workers.stt_pool import transcribe
from app.workers.tts_pool import synthesize

logger = get_logger(__name__)


class SpeechPipeline:
    def __init__(self, session: CallSession) -> None:
        self.session = session

    async def process_audio_chunk(self, audio_chunk: bytes) -> Optional[bytes]:
        self.session.add_speech_chunk(audio_chunk)

        should_flush = (
            self.session.speech_chunk_counter % STT_FLUSH_EVERY_N_CHUNKS == 0
            or self.session.seconds_since_last_flush() >= STT_FLUSH_INTERVAL_SECONDS
        )
        if not should_flush:
            return None

        return await self._flush_buffer("Dispatching buffered audio to STT")

    async def flush_on_speech_stop(self) -> Optional[bytes]:
        return await self._flush_buffer("Speech stop detected, flushing buffered audio to STT")

    async def flush_remaining(self) -> Optional[bytes]:
        return await self._flush_buffer("Flushing remaining buffered audio to STT")

    async def _flush_buffer(self, reason: str) -> Optional[bytes]:
        buffered_audio = self.session.pop_buffer()
        if not buffered_audio:
            return None

        logger.info(
            "%s: speech_chunks=%s bytes=%s",
            reason,
            self.session.speech_chunk_counter,
            len(buffered_audio),
        )
        text = await transcribe(buffered_audio)
        if not text:
            logger.info("STT returned no text")
            return None

        await self._append_transcript(text)
        logger.info("STT recognized text: %s", text)

        logger.info("Dispatching text to TTS: %s", text)
        return await synthesize(text)

    async def _append_transcript(self, text: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {text}\n"

        def _write() -> None:
            TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRANSCRIPT_PATH.open("a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_write)
        logger.info("Transcript appended to %s", TRANSCRIPT_PATH)
