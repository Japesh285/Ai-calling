from pathlib import Path

# STT worker endpoint (Faster-Whisper service)
STT_WORKERS = [
    "http://localhost:8001/transcribe",
]
TTS_WORKERS = [
    "http://localhost:8002/synthesize",
]
AI_BRAIN_VOICE_URL = "http://127.0.0.1:9000/voice"

# Audio buffering / flush policy (gateway-side orchestration only)
STT_FLUSH_EVERY_N_CHUNKS = 250
STT_FLUSH_INTERVAL_SECONDS = 5.0
PCM_SAMPLE_RATE = 8000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2
PCMA_FRAME_BYTES = 160

# Realtime inbound audio and voice segmentation
PCM_FRAME_BYTES = 320
FRAME_DURATION_MS = 20
SPEECH_START_RMS_THRESHOLD = 250
SPEECH_STOP_RMS_THRESHOLD = 150
SILENCE_STOP_FRAMES = 20
SILENCE_TIMEOUT_MS = 400
MAX_SPEECH_LENGTH_MS = 10_000
MIN_SPEECH_LENGTH_MS = 300
TTS_PLAYBACK_GUARD_MS = 1200

# Outbound TTS back to FreeSWITCH
FREESWITCH_OUTBOUND_MODE = "binary"

# Local paths
BASE_DIR = Path(__file__).resolve().parents[2]

# Transcript output file
TRANSCRIPT_PATH = BASE_DIR / "logs" / "stt_output.log"
AI_BRAIN_RESPONSE_PATH = BASE_DIR / "logs" / "brain_responses.txt"
FREESWITCH_AUDIO_DUMP_DIR = BASE_DIR / "logs" / "freeswitch_audio"
FREESWITCH_RAW_DUMP_DIR = BASE_DIR / "logs" / "freeswitch_raw"
TTS_AUDIO_DUMP_DIR = BASE_DIR / "logs" / "tts_audio"
