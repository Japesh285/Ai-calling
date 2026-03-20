import os
from pathlib import Path

# STT worker endpoint (Faster-Whisper service)
STT_WORKERS = [
    "http://localhost:8001/transcribe",
]
# TTS worker endpoint (Indic Parler TTS with Delhi accent)
TTS_WORKERS = [
    "http://localhost:8002/synthesize",
]
# Indic Parler (or compatible) — streaming + blocking synthesize
# Default 8002 matches a typical `uvicorn indic_server:app --port 8002` setup.
INDIC_TTS_URL = os.environ.get("INDIC_TTS_URL", "http://127.0.0.1:8002")
AI_BRAIN_VOICE_URL = "http://127.0.0.1:9000/voice"

# Raw RTP (symmetric) — advertise this IP:port to FreeSWITCH / your RTP forwarder
RTP_ADVERTISE_HOST = os.environ.get("RTP_ADVERTISE_HOST", "127.0.0.1")
# Outbound RTP payload type: 8 = PCMA (A-law), 0 = PCMU — match your FS leg codec
RTP_OUT_PAYLOAD_TYPE = int(os.environ.get("RTP_OUT_PAYLOAD_TYPE", "8"))
# RTP playout: fill this much 8 kHz PCM (mono16) before first packet — absorbs bursty TTS/HTTP.
RTP_PLAYOUT_PREBUFFER_MS = int(os.environ.get("RTP_PLAYOUT_PREBUFFER_MS", "400"))

# Audio buffering / flush policy (gateway-side orchestration only)
STT_FLUSH_EVERY_N_CHUNKS = 250
STT_FLUSH_INTERVAL_SECONDS = 0.8
PCM_SAMPLE_RATE = 8000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2
PCMA_FRAME_BYTES = 160

# Realtime inbound audio and voice segmentation
PCM_FRAME_BYTES = 320
FRAME_DURATION_MS = 20
SPEECH_START_RMS_THRESHOLD = 400      # Reject background noise (was 250)
SPEECH_STOP_RMS_THRESHOLD = 200       # Faster silence detection (was 150)
SILENCE_STOP_FRAMES = 6               # 120ms silence = stop (was 10 = 200ms)
SILENCE_TIMEOUT_MS = 300
MAX_SPEECH_LENGTH_MS = 10_000
MIN_SPEECH_LENGTH_MS = 500            # Require 10+ frames of speech (was 300)
TTS_PLAYBACK_GUARD_MS = 600

# Inbound RMS must exceed this to count as barge-in while AI audio is playing
# (echo/room noise often crosses SPEECH_START_RMS_THRESHOLD but stays below this).
# Raised from 950 → 1080 to reduce false interrupts from speaker echo on telephony.
BARGE_IN_RMS_THRESHOLD = 1080

# Log full STT JSON from 8001 when transcript is empty or filtered (debug garbled Hindi).
STT_LOG_RAW_ON_FAILURE = True

# Outbound TTS back to FreeSWITCH
FREESWITCH_OUTBOUND_MODE = "binary"

# When True: stream PCM through a named pipe (one uuid_broadcast per utterance).
# When False: write temp .wav per segment + uuid_broadcast (most reliable on FreeSWITCH).
# FIFO + raw .r8 on a pipe is fragile (non-blocking write drops, FS may not seek pipes).
USE_FIFO_PLAYBACK = True

# Local paths
BASE_DIR = Path(__file__).resolve().parents[2]

# Transcript output file
TRANSCRIPT_PATH = BASE_DIR / "logs" / "stt_output.log"
AI_BRAIN_RESPONSE_PATH = BASE_DIR / "logs" / "brain_responses.txt"
FREESWITCH_AUDIO_DUMP_DIR = BASE_DIR / "logs" / "freeswitch_audio"
FREESWITCH_RAW_DUMP_DIR = BASE_DIR / "logs" / "freeswitch_raw"
TTS_AUDIO_DUMP_DIR = BASE_DIR / "logs" / "tts_audio"
