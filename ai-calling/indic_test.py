import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ai4bharat/indic-parler-tts"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)

print("Model loaded.")

description = "A clear female voice speaking Hindi."
prompt = "नमस्ते आपका स्वागत है"

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print("Generating audio...")

generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_ids
)

audio = generation.cpu().numpy().squeeze()

sf.write("output.wav", audio, 24000)

print("Done! Audio saved as output.wav")