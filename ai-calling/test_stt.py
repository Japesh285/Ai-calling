from __future__ import annotations

import sys
import wave
from pathlib import Path

from app.stt.faster_whisper_stt import FasterWhisperSTT


def load_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit PCM")
        if wav_file.getframerate() != 8000:
            raise ValueError("WAV must be 8000 Hz")
        return wav_file.readframes(wav_file.getnframes())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python test_stt.py <wav-file>")

    wav_path = Path(sys.argv[1]).resolve()
    audio_bytes = load_wav_pcm(wav_path)

    stt = FasterWhisperSTT()
    transcript = stt.transcribe(audio_bytes)

    print(f"Detected language: {stt.last_detected_language}")
    print(f"Transcript: {transcript}")


if __name__ == "__main__":
    main()
