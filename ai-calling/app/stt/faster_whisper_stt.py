from __future__ import annotations

import re

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
            "large-v3",
            device=device,
            compute_type=compute_type,
        )
        self.last_detected_language = ""

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_8k = np.frombuffer(audio_bytes, dtype=np.int16)
        logger.info("STT audio length: samples=%s bytes=%s", audio_8k.size, len(audio_bytes))

        audio_float32 = audio_8k.astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio_float32, orig_sr=8000, target_sr=16000)

        segments, info = self.model.transcribe(
        audio_16k,
        language=None,              # allow auto detection
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
