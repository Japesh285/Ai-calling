import asyncio
import io
import threading
import wave
from math import gcd

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import resample_poly
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig


XTTS_SAMPLE_RATE = 24000
DEFAULT_SPEAKER_WAV = "/home/developer/ai-calling/FLN Vo (1).wav"
# XTTS v2 expects reference audio at 22050 Hz for best voice conditioning
XTTS_REF_SAMPLE_RATE = 22050


def _prepare_reference_audio(src_path: str) -> str:
    """
    XTTS v2 voices conditioned on 22050 Hz audio. The reference file is 16kHz,
    so we upsample it once at startup using scipy (anti-aliased) and cache it.
    Returns the path of the correctly-sampled reference file.
    """
    out_path = src_path.replace(".wav", "_22050.wav")
    try:
        data, sr = sf.read(src_path, dtype="int16", always_2d=False)
        if sr != XTTS_REF_SAMPLE_RATE:
            g = gcd(sr, XTTS_REF_SAMPLE_RATE)
            up = XTTS_REF_SAMPLE_RATE // g
            down = sr // g
            resampled = resample_poly(data.astype(np.float32), up, down)
            data = np.clip(resampled, -32768, 32767).astype(np.int16)
        sf.write(out_path, data, XTTS_REF_SAMPLE_RATE, subtype="PCM_16")
        print(f"Reference audio prepared at {XTTS_REF_SAMPLE_RATE} Hz → {out_path}")
    except Exception as e:
        print(f"Reference audio preparation failed, using original: {e}")
        return src_path
    return out_path


def _load_xtts() -> Xtts:
    from pathlib import Path
    from TTS.utils.generic_utils import get_user_data_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve model directory directly — ModelManager.download_model() returns
    # None for already-downloaded models in some Coqui TTS versions.
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = Path(get_user_data_dir("tts")) / "tts" / model_name.replace("/", "--")
    model_path = str(model_dir)
    config_path = str(model_dir / "config.json")

    with torch.serialization.safe_globals(
        [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
    ):
        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        model.to(device)
    return model


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"


app = FastAPI(title="TTS Worker")

# Module-level state
_xtts_model: Xtts | None = None
_gpt_cond_latent = None
_speaker_embedding = None
_ref_audio_path: str = ""
_model_lock = threading.Lock()


def _init_model_and_latents() -> None:
    global _xtts_model, _gpt_cond_latent, _speaker_embedding, _ref_audio_path
    try:
        ref_path = _prepare_reference_audio(DEFAULT_SPEAKER_WAV)
        _ref_audio_path = ref_path

        model = _load_xtts()
        _xtts_model = model

        # This version of Coqui TTS ships without 'hi' in tokenizer.char_limits,
        # despite the model supporting Hindi. Patch it so split_sentence() doesn't
        # raise KeyError on every Hindi synthesis request.
        # 150 chars (not 250) because Devanagari characters are phonetically denser —
        # each akshara represents a full syllable, so 150 Devanagari chars ≈ 250 Latin chars.
        if "hi" not in model.tokenizer.char_limits:
            model.tokenizer.char_limits["hi"] = 150
            print("Patched tokenizer: added char_limits['hi'] = 150")

        # Compute and cache conditioning latents once — reused for every synthesis call.
        # gpt_cond_len=30 / max_ref_length=30 means we use the full 19-second reference.
        # gpt_cond_chunk_len=6 (XTTS default) chunks the reference into 6-second windows
        # for GPT conditioning — longer chunks = stronger speaker identity capture.
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[ref_path],
            gpt_cond_len=30,
            max_ref_length=30,
            gpt_cond_chunk_len=6,
            sound_norm_refs=False,
        )
        _gpt_cond_latent = gpt_cond_latent
        _speaker_embedding = speaker_embedding
        print("XTTS model and voice latents ready")
    except Exception as e:
        print(f"XTTS model init failed: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(_init_model_and_latents)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "tts worker running"}


def _synthesize_wav_bytes(text: str, language: str) -> bytes:
    if _xtts_model is None or _gpt_cond_latent is None or _speaker_embedding is None:
        raise RuntimeError("TTS model not yet initialised")

    with _model_lock:
        # repetition_penalty=2.0 for Hindi: Hindi has many naturally repeating phoneme
        # patterns (matras, conjuncts). A value above 3.0 forces XTTS to avoid these
        # natural repetitions and can select phonemes from other language patterns in its
        # multilingual vocabulary — producing Arabic/Persian-sounding output.
        rep_penalty = 2.0 if language == "hi" else 2.0
        # Slightly higher temperature for Hindi gives better phoneme variety
        temp = 0.75 if language == "hi" else 0.75

        out = _xtts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=_gpt_cond_latent,
            speaker_embedding=_speaker_embedding,
            speed=1.1 if language == "hi" else 1.0,   # North Indian Hindi is faster-paced
            temperature=temp,
            repetition_penalty=rep_penalty,
            top_k=50,
            top_p=0.85,
            enable_text_splitting=True,
        )

    wav_tensor = out["wav"]
    if isinstance(wav_tensor, torch.Tensor):
        wav_np = wav_tensor.squeeze().cpu().numpy()
    else:
        wav_np = np.array(wav_tensor, dtype=np.float32)

    # Clip and convert to int16 PCM
    wav_int16 = np.clip(wav_np * 32767, -32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(XTTS_SAMPLE_RATE)
        wf.writeframes(wav_int16.tobytes())
    return buffer.getvalue()


@app.post("/synthesize")
async def synthesize(payload: SynthesizeRequest) -> Response:
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is required")

    wav_bytes = await asyncio.to_thread(
        _synthesize_wav_bytes,
        clean_text,
        payload.language,
    )
    if not wav_bytes:
        raise HTTPException(status_code=500, detail="TTS generated no audio")

    return Response(content=wav_bytes, media_type="audio/wav")
