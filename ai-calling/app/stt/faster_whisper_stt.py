from __future__ import annotations

import re
from collections.abc import Generator

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from unidecode import unidecode

from app.utils.logger import get_logger


logger = get_logger(__name__)


class FasterWhisperSTT:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            "medium",
            device=device,
            compute_type=compute_type,
        )
        self.last_detected_language = ""
        self._stream_buffer: np.ndarray | None = None
        self._stream_language = ""

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_8k = np.frombuffer(audio_bytes, dtype=np.int16)
        logger.info("STT audio length: samples=%s bytes=%s", audio_8k.size, len(audio_bytes))

        audio_float32 = audio_8k.astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio_float32, orig_sr=8000, target_sr=16000)

        segments, info = self.model.transcribe(
            audio_16k,
            language=None,
            task="transcribe",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt="The following conversation may contain Hindi and English spoken by an Indian speaker.",
        )

        self.last_detected_language = getattr(info, "language", "") or ""
        if self.last_detected_language not in {"en", "hi"}:
            logger.warning("Unexpected language detected: %s → forcing hi", self.last_detected_language)
            self.last_detected_language = "hi"
        logger.info("Detected language: %s", self.last_detected_language)

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        text = unidecode(text)
        text = re.sub(r"\s+", " ", text).lower().strip()

        logger.info("Transcript result: %s", text)
        return text

    def init_stream(self) -> None:
        """Initialize a new streaming session."""
        self._stream_buffer = None
        self._stream_language = ""

    def stream_transcribe(self, audio_chunk: bytes) -> str:
        """
        Process an audio chunk and return incremental transcription.
        Accumulates audio until we have enough for transcription.
        """
        audio_8k = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_8k.astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio_float32, orig_sr=8000, target_sr=16000)

        if self._stream_buffer is None:
            self._stream_buffer = audio_16k
        else:
            self._stream_buffer = np.concatenate([self._stream_buffer, audio_16k])

        # Only transcribe if we have at least 1 second of audio
        if len(self._stream_buffer) < 16000:
            return ""

        segments, info = self.model.transcribe(
            self._stream_buffer,
            language=self._stream_language if self._stream_language else None,
            task="transcribe",
            beam_size=1,
            condition_on_previous_text=True,
            vad_filter=False,
            initial_prompt="The following conversation may contain Hindi and English spoken by an Indian speaker.",
            without_timestamps=True,
        )

        self._stream_language = getattr(info, "language", "") or ""
        if self._stream_language not in {"en", "hi"}:
            self._stream_language = "hi"

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        text = unidecode(text)
        text = re.sub(r"\s+", " ", text).lower().strip()

        # Keep last 2 seconds for context continuity
        keep_samples = int(32000)  # 2 seconds at 16kHz
        if len(self._stream_buffer) > keep_samples:
            self._stream_buffer = self._stream_buffer[-keep_samples:]

        return text

    def finalize_stream(self) -> str:
        """Finalize the streaming session and return final transcription."""
        if self._stream_buffer is None or len(self._stream_buffer) == 0:
            return ""

        segments, info = self.model.transcribe(
            self._stream_buffer,
            language=self._stream_language if self._stream_language else None,
            task="transcribe",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt="The following conversation may contain Hindi and English spoken by an Indian speaker.",
        )

        self.last_detected_language = getattr(info, "language", "") or ""
        if self.last_detected_language not in {"en", "hi"}:
            logger.warning("Unexpected language detected: %s → forcing hi", self.last_detected_language)
            self.last_detected_language = "hi"

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        text = unidecode(text)
        text = re.sub(r"\s+", " ", text).lower().strip()

        self._stream_buffer = None
        return text
