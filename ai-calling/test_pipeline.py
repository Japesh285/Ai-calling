from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

from app.pipeline.speech_pipeline import ask_brain, transcribe_audio


def load_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit PCM")
        if wav_file.getframerate() != 8000:
            raise ValueError("WAV must be 8000 Hz")
        return wav_file.readframes(wav_file.getnframes())


async def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: ./esp/bin/python test_pipeline.py <wav-file>")

    wav_path = Path(sys.argv[1]).resolve()
    audio_bytes = load_wav_pcm(wav_path)

    transcript = await transcribe_audio(audio_bytes)
    print(f"User: {transcript}")

    response = await ask_brain(transcript)
    print(f"AI: {response}")


if __name__ == "__main__":
    asyncio.run(main())
