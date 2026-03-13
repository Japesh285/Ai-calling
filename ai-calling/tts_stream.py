import io
import sys

import torch
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig


def _load_tts() -> TTS:
    # XTTS uses custom checkpoint classes that must be allow-listed for torch loading.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.serialization.safe_globals(
        [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
    ):
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def main() -> int:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print("Missing text input", file=sys.stderr)
        return 1

    tts = _load_tts()
    wav = tts.tts(
        text=text,
        speaker="Ana Florence",
        language="en",
    )

    buffer = io.BytesIO()
    tts.synthesizer.save_wav(wav=wav, path=buffer)
    sys.stdout.buffer.write(buffer.getvalue())
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
