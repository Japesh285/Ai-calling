from pathlib import Path

# STT worker endpoint (Faster-Whisper service)
STT_WORKERS = [
    "http://localhost:8001/transcribe",
]

# Audio buffering / flush policy (gateway-side orchestration only)
STT_FLUSH_EVERY_N_CHUNKS = 250
STT_FLUSH_INTERVAL_SECONDS = 5.0
PCM_SAMPLE_RATE = 8000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH_BYTES = 2

# Realtime inbound audio and VAD
PCMA_FRAME_BYTES = 320
VAD_RMS_THRESHOLD = 450
VAD_USE_WEBRTC = False
VAD_WEBRTC_AGGRESSIVENESS = 2
VAD_SUBFRAME_MS = 20
VAD_SPEECH_STOP_MS = 200
VAD_PREROLL_MS = 200

# Transcript output file
BASE_DIR = Path(__file__).resolve().parents[2]
TRANSCRIPT_PATH = BASE_DIR / "logs" / "stt_output.log"
FREESWITCH_AUDIO_DUMP_DIR = BASE_DIR / "logs" / "freeswitch_audio"
FREESWITCH_RAW_DUMP_DIR = BASE_DIR / "logs" / "freeswitch_raw"
TTS_AUDIO_DUMP_DIR = BASE_DIR / "logs" / "tts_audio"

# TTS script execution (external worker script, no in-process model loading)
TTS_SCRIPT_CWD = BASE_DIR
TTS_SCRIPT_NAME = "tts_stream.py"
TTS_COMMAND = ["/home/developer/ai-calling/venv/bin/python", TTS_SCRIPT_NAME]
