from __future__ import annotations

import audioop
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

from app.config.settings import (
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    VAD_PREROLL_MS,
    VAD_RMS_THRESHOLD,
    VAD_SPEECH_STOP_MS,
    VAD_SUBFRAME_MS,
    VAD_USE_WEBRTC,
    VAD_WEBRTC_AGGRESSIVENESS,
)

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    webrtcvad = None


@dataclass(slots=True)
class VADResult:
    pcm16: bytes
    rms: int
    is_speech: bool
    speech_started: bool = False
    speech_stopped: bool = False
    speech_chunk: bytes = b""


class RealtimeSpeechGate:
    def __init__(self) -> None:
        self._bytes_per_ms = (
            PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
        ) // 1000
        self._subframe_bytes = self._bytes_per_ms * VAD_SUBFRAME_MS
        self._stop_frames = max(1, VAD_SPEECH_STOP_MS // VAD_SUBFRAME_MS)
        self._preroll_frames = max(0, VAD_PREROLL_MS // VAD_SUBFRAME_MS)
        self._preroll: Deque[bytes] = deque(maxlen=self._preroll_frames)
        self._is_speaking = False
        self._silence_frames = 0
        self._vad = self._build_vad()

    def process_frame(self, pcm16: bytes) -> VADResult:
        rms = audioop.rms(pcm16, PCM_SAMPLE_WIDTH_BYTES) if pcm16 else 0
        is_speech = self._detect_speech(pcm16, rms)

        speech_started = False
        speech_stopped = False
        speech_chunk = b""

        if is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                speech_started = True
                if self._preroll:
                    speech_chunk += b"".join(self._preroll)
            self._silence_frames = 0
            speech_chunk += pcm16
        else:
            self._preroll.append(pcm16)
            if self._is_speaking:
                self._silence_frames += max(1, len(pcm16) // self._subframe_bytes)
                if self._silence_frames >= self._stop_frames:
                    self._is_speaking = False
                    self._silence_frames = 0
                    speech_stopped = True
            else:
                self._silence_frames = 0

        if is_speech:
            for chunk in self._split_subframes(pcm16):
                self._preroll.append(chunk)

        return VADResult(
            pcm16=pcm16,
            rms=rms,
            is_speech=is_speech,
            speech_started=speech_started,
            speech_stopped=speech_stopped,
            speech_chunk=speech_chunk,
        )

    def _build_vad(self):
        if not VAD_USE_WEBRTC or webrtcvad is None:
            return None

        vad = webrtcvad.Vad()
        vad.set_mode(VAD_WEBRTC_AGGRESSIVENESS)
        return vad

    def _detect_speech(self, pcm16: bytes, rms: int) -> bool:
        if rms < VAD_RMS_THRESHOLD:
            return False

        if self._vad is None:
            return True

        speech_votes = 0
        total_votes = 0
        for chunk in self._split_subframes(pcm16):
            if len(chunk) != self._subframe_bytes:
                continue
            total_votes += 1
            if self._vad.is_speech(chunk, PCM_SAMPLE_RATE):
                speech_votes += 1
        if total_votes == 0:
            return False
        return speech_votes >= max(1, (total_votes + 1) // 2)

    def _split_subframes(self, pcm16: bytes) -> Iterable[bytes]:
        if self._subframe_bytes <= 0:
            return []
        return (
            pcm16[index : index + self._subframe_bytes]
            for index in range(0, len(pcm16), self._subframe_bytes)
        )
