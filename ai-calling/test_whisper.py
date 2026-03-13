from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe("/home/developer/ai-calling/harvard.wav")

print("Detected language:", info.language)

for segment in segments:
    print(segment.text)