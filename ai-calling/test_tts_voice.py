import torch

# ==== PATCH (IMPORTANT) ====
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load
# ==========================

import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

MODEL_DIR = "/home/developer/hindi-tts-finetuned"
CONFIG_PATH = f"{MODEL_DIR}/config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.serialization.safe_globals(
    [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
):
    config = XttsConfig()
    config.load_json(CONFIG_PATH)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)
    model.to(device)

print("✅ Model loaded")

# Fix Hindi tokenizer
if "hi" not in model.tokenizer.char_limits:
    model.tokenizer.char_limits["hi"] = 150

text = "नमस्ते, यह एक टेस्ट है।"

out = model.inference(
    text=text,
    language="hi",
)

sf.write("output.wav", out["wav"], 24000)

print("✅ Done")