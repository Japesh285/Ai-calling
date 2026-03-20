"""
Indic TTS Server using AI4Bharat Parler TTS
Exact clone of tts_server.py structure with Indic Parler TTS model

Supports:
- /synthesize (blocking, complete WAV)
- /synthesize/stream (streaming, chunks as they're generated)
"""
import asyncio
import io
import os
import struct
import threading
import wave
from math import gcd
from threading import Thread

# Set transformers to offline mode (no internet required after first download)
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from scipy.signal import resample_poly

from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig, ParlerTTSStreamer
from transformers import AutoTokenizer, AutoFeatureExtractor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Register parler_tts config (required for Auto classes to work)
if 'parler_tts' not in CONFIG_MAPPING:
    CONFIG_MAPPING.register('parler_tts', ParlerTTSConfig, exist_ok=True)

MODEL_REPO = "ai4bharat/indic-parler-tts"
DEFAULT_SPEAKER_WAV = "/home/developer/ai-calling/FLN Vo (1).wav"
# Match XTTS output: 24kHz sample rate for consistent audio pipeline
DEFAULT_SAMPLE_RATE = 24000
# Streaming: fixed packet duration for downstream jitter buffers (8 kHz telephony = 20 ms; at 24 kHz = 480 samples).
_STREAM_PACKET_MS = int(os.environ.get("INDIC_STREAM_PACKET_MS", "20"))
_STREAM_PREBUFFER_MS = int(os.environ.get("INDIC_STREAM_PREBUFFER_MS", "20"))
_STREAM_SAMPLES_PER_PACKET = max(1, int(DEFAULT_SAMPLE_RATE * _STREAM_PACKET_MS / 1000))
_STREAM_PREBUFFER_SAMPLES = max(_STREAM_SAMPLES_PER_PACKET, int(DEFAULT_SAMPLE_RATE * _STREAM_PREBUFFER_MS / 1000))
# Match XTTS voice conditioning settings
# Delhi Hindi accent for natural Indian voice
VOICE_DESCRIPTION = "A female speaker with a Delhi Hindi accent speaks very quickly, briskly, and energetically while staying clear and intelligible. The recording is high quality with a warm, friendly tone typical of North Indian Hindi speech."
TTS_SPEED_FACTOR = 1.05

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


def _build_wav_header(sample_rate: int = 24000) -> bytes:
    """
    Build minimal WAV header (44 bytes) for streaming.
    Uses placeholder sizes for streaming compatibility.
    """
    # WAV header format for streaming
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        0xFFFFFFFF,  # File size (placeholder - streaming)
        b'WAVE',
        b'fmt ',
        16,          # Subchunk1Size (16 for PCM)
        1,           # AudioFormat (1 = PCM)
        1,           # NumChannels (mono)
        sample_rate, # SampleRate
        sample_rate * 2,  # ByteRate (sample_rate * num_channels * bits_per_sample/8)
        2,           # BlockAlign (num_channels * bits_per_sample/8)
        16,          # BitsPerSample
        b'data',
        0xFFFFFFFF,  # Subchunk2Size (placeholder)
    )
    return header


def _trim_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.003,
    min_silence_ms: int = 250,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    window_size = max(1, int(sample_rate * (min_silence_ms / 1000.0)))
    abs_audio = np.abs(audio)

    last_non_silent = len(audio)
    for index in range(len(audio), 0, -window_size):
        window_start = max(0, index - window_size)
        if np.max(abs_audio[window_start:index]) > silence_threshold:
            last_non_silent = index
            break

    return audio[:last_non_silent]


def _speed_up_audio(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    if audio.size == 0 or speed_factor <= 1.0:
        return audio

    speed_factor = float(speed_factor)
    up = 100
    down = max(1, int(round(speed_factor * up)))
    sped_up = resample_poly(audio, up, down).astype(np.float32)

    peak = float(np.max(np.abs(sped_up))) if sped_up.size else 0.0
    if peak > 0.98:
        sped_up = sped_up * (0.98 / peak)

    return sped_up


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float audio in [-1, 1] to PCM16 bytes."""
    if audio.size == 0:
        return b""
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()


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
    audio = _speed_up_audio(audio, TTS_SPEED_FACTOR)
    audio = _trim_trailing_silence(audio, 24000, silence_threshold=0.004, min_silence_ms=180)

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


@app.post("/synthesize/stream")
async def synthesize_stream(payload: SynthesizeRequest) -> StreamingResponse:
    """
    Stream TTS audio in small chunks as they're generated.
    
    Returns WAV stream:
    - First bytes: WAV header (44 bytes)
    - Then PCM16 mono @ 24 kHz in packets of INDIC_STREAM_PACKET_MS (default 20 ms = 960 bytes).
    - Optional INDIC_STREAM_PREBUFFER_MS (default 20 ms) delays first audio until that much PCM is ready.
    """
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

    return StreamingResponse(
        _generate_wav_chunks(clean_text, lang),
        media_type="audio/wav"
    )


async def _generate_wav_chunks(text: str, language: str):
    """Generator that yields WAV header then audio chunks as they're generated."""
    if _model is None or _tokenizer is None or _description_tokenizer is None:
        raise RuntimeError("TTS model not yet initialised")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_sample_rate = getattr(_model.config, 'sampling_rate', 24000)

    # Setup inputs
    description_input_ids = _description_tokenizer(VOICE_DESCRIPTION, return_tensors="pt").to(device)
    prompt_input_ids = _tokenizer(text, return_tensors="pt").to(device)
    estimated_audio_tokens = prompt_input_ids.input_ids.shape[1] * 15
    max_length = min(estimated_audio_tokens + 50, 1500)

    # Create streamer - yields ~0.5s chunks (50 tokens * ~11ms per token)
    streamer = ParlerTTSStreamer(
        _model,
        device=device,
        play_steps=40,
        stride=5,  # Overlap for smoother playback
    )

    generation_kwargs = dict(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
        streamer=streamer,
        pad_token_id=_tokenizer.pad_token_id,
        max_length=max_length,
        min_new_tokens=8,
        do_sample=True,
        temperature=0.75,
    )

    if _ref_inputs is not None:
        generation_kwargs.update(_ref_inputs)

    # Yield WAV header first
    yield _build_wav_header(sample_rate=24000)

    # Run generation in background thread
    def run_generation():
        with torch.inference_mode():
            _model.generate(**generation_kwargs)
            streamer.end()

    thread = Thread(target=run_generation)
    thread.start()

    flush_chunk_samples = _STREAM_SAMPLES_PER_PACKET
    prebuffer_samples = _STREAM_PREBUFFER_SAMPLES
    stream_started = False

    pending_audio = np.array([], dtype=np.float32)
    try:
        for chunk in streamer:
            chunk_audio = np.asarray(chunk, dtype=np.float32)
            if model_sample_rate != DEFAULT_SAMPLE_RATE:
                chunk_audio = _resample_audio(chunk_audio, model_sample_rate, DEFAULT_SAMPLE_RATE)
            pending_audio = np.concatenate([pending_audio, chunk_audio])

            while pending_audio.size >= flush_chunk_samples:
                if not stream_started and pending_audio.size < prebuffer_samples:
                    break
                stream_started = True
                emit_chunk = pending_audio[:flush_chunk_samples].copy()
                pending_audio = pending_audio[flush_chunk_samples:]
                emit_chunk = _speed_up_audio(emit_chunk, TTS_SPEED_FACTOR)
                yield _pcm16_bytes(emit_chunk)
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        thread.join(timeout=10.0)

    pending_audio = _trim_trailing_silence(pending_audio, DEFAULT_SAMPLE_RATE)
    while pending_audio.size >= flush_chunk_samples:
        emit_chunk = pending_audio[:flush_chunk_samples].copy()
        pending_audio = pending_audio[flush_chunk_samples:]
        emit_chunk = _speed_up_audio(emit_chunk, TTS_SPEED_FACTOR)
        yield _pcm16_bytes(emit_chunk)
    if pending_audio.size > 0:
        yield _pcm16_bytes(_speed_up_audio(pending_audio, TTS_SPEED_FACTOR))


if __name__ == "__main__":
    import uvicorn

    _port = int(os.environ.get("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=_port)
