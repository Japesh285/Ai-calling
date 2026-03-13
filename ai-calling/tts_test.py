import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

from TTS.api import TTS

print("Loading XTTS...")

# Allow XTTS checkpoint classes in one block
with torch.serialization.safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig
]):

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

print("Model loaded")

tts.tts_to_file(
    text="Hello namaste kaise ho, this is the AI calling assistant.",
    speaker="Ana Florence",
    language="en",
    file_path="voice.wav"
)

print("Done")