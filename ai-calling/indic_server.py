"""
Indic TTS Server using AI4Bharat Parler TTS
Exact clone of tts_server.py structure with Indic Parler TTS model
"""
import asyncio
import io
import os
import threading
import wave
from math import gcd

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import resample_poly

from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer, AutoFeatureExtractor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Register parler_tts config (required for Auto classes to work)
if 'parler_tts' not in CONFIG_MAPPING:
    CONFIG_MAPPING.register('parler_tts', ParlerTTSConfig, exist_ok=True)

MODEL_REPO = "ai4bharat/indic-parler-tts"
DEFAULT_SPEAKER_WAV = "/home/developer/ai-calling/FLN Vo (1).wav"
# Match XTTS output: 24kHz sample rate for consistent audio pipeline
DEFAULT_SAMPLE_RATE = 24000
# Match XTTS voice conditioning settings
VOICE_DESCRIPTION = "A female speaker delivers a clear and friendly speech with moderate speed and pitch. The recording is of high quality, with the speaker's voice sounding natural and warm."

# Language mapping for Indic languages
INDIC_LANGUAGES = {
    "hi": "hindi",
    "bn": "bengali",
    "te": "telugu",
    "mr": "marathi",
    "ta": "tamil",
    "gu": "gujarati",
    "kn": "kannada",
    "pa": "punjabi",
    "ml": "malayalam",
    "or": "odia",
}


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "hi"


app = FastAPI(title="Indic TTS Worker")

# Module-level state
_model: ParlerTTSForConditionalGeneration | None = None
_tokenizer: AutoTokenizer | None = None
_description_tokenizer: AutoTokenizer | None = None
_feature_extractor: AutoFeatureExtractor | None = None
_ref_inputs: dict | None = None
_model_lock = threading.Lock()


def _load_model_and_tokenizers() -> None:
    global _model, _tokenizer, _description_tokenizer, _feature_extractor, _ref_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Indic Parler TTS model on {device}...")

    # Load model
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        local_files_only=True
    )

    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path,
        trust_remote_code=True,
        local_files_only=True
    )

    # Prepare reference audio if available (match indic_test.py pattern)
    feature_extractor = None
    ref_inputs = None
    if os.path.exists(DEFAULT_SPEAKER_WAV):
        print(f"Loading reference audio: {DEFAULT_SPEAKER_WAV}")
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                MODEL_REPO,
                trust_remote_code=True,
                local_files_only=True
            )
            import librosa
            ref_audio, ref_sr = librosa.load(
                DEFAULT_SPEAKER_WAV,
                sr=feature_extractor.sampling_rate
            )
            ref_inputs = feature_extractor(
                ref_audio,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt"
            ).to(device)
            print("Reference audio loaded for voice cloning")
        except Exception as e:
            print(f"Reference audio loading failed: {e}")
            print("Will use default voice")
    else:
        print("No reference audio found, will use default voice")

    _model = model
    _tokenizer = tokenizer
    _description_tokenizer = description_tokenizer
    _feature_extractor = feature_extractor
    _ref_inputs = ref_inputs

    print("Indic Parler TTS model and tokenizers ready")


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to match XTTS output sample rate (24kHz)."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def _synthesize_wav_bytes(text: str, language: str) -> bytes:
    if _model is None or _tokenizer is None or _description_tokenizer is None:
        raise RuntimeError("TTS model not yet initialised")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with _model_lock:
        # Tokenize description (voice characteristics) - goes to text encoder
        description_input_ids = _description_tokenizer(VOICE_DESCRIPTION, return_tensors="pt").to(device)

        # Tokenize prompt (text to synthesize) - goes to prompt encoder
        prompt_input_ids = _tokenizer(text, return_tensors="pt").to(device)

        # Calculate appropriate max_length based on input text length
        estimated_audio_tokens = prompt_input_ids.input_ids.shape[1] * 15
        max_length = min(estimated_audio_tokens + 50, 1500)

        # Language-specific temperature (match XTTS settings)
        temp = 0.75 if language == "hi" else 0.75

        # Generate audio with Parler TTS (match indic_test.py pattern)
        with torch.inference_mode():
            if _ref_inputs is not None:
                # Voice cloning mode
                generation = _model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    **_ref_inputs,
                    pad_token_id=_tokenizer.pad_token_id,
                    max_length=max_length,
                    min_new_tokens=10,
                    do_sample=True,
                    temperature=temp,
                )
            else:
                # Standard TTS mode
                generation = _model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    pad_token_id=_tokenizer.pad_token_id,
                    max_length=max_length,
                    min_new_tokens=10,
                    do_sample=True,
                    temperature=temp,
                )

        # Convert to numpy (match indic_test.py pattern)
        if hasattr(generation, 'audio_values'):
            audio = generation.audio_values.cpu().numpy()[0]
        else:
            audio = generation.cpu().numpy()[0]

    # Get model's native sample rate
    model_sample_rate = getattr(_model.config, 'sampling_rate', 24000)

    # Resample to 24kHz to match tts_server.py output
    audio = _resample_audio(audio, model_sample_rate, 24000)

    # Clip and convert to int16 PCM (EXACT match to tts_server.py)
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # Write WAV buffer (EXACT match to tts_server.py)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(_load_model_and_tokenizers)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "indic tts worker running"}


@app.post("/synthesize")
async def synthesize(payload: SynthesizeRequest) -> Response:
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Validate language
    lang = payload.language.lower()
    if lang not in INDIC_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {lang}. Supported: {list(INDIC_LANGUAGES.keys())}"
        )

    wav_bytes = await asyncio.to_thread(
        _synthesize_wav_bytes,
        clean_text,
        lang,
    )
    if not wav_bytes:
        raise HTTPException(status_code=500, detail="TTS generated no audio")

    return Response(content=wav_bytes, media_type="audio/wav")
