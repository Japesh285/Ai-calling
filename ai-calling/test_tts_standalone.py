#!/usr/bin/env python3
"""
Standalone TTS Voice Testing Script

Tests voice synthesis using the same logic as tts_server.py.
Generates multiple audio samples for quality evaluation.

Usage:
    python test_tts_standalone.py
    python test_tts_standalone.py --compare-speeds
    python test_tts_standalone.py --custom-text "Your custom text here"
"""

# PyTorch 2.6+ compatibility: patch torch.load BEFORE any other imports
import torch
import torch.serialization

_orig_load = torch.serialization.load

def _load_with_weights_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.serialization.load = _load_with_weights_false
# Also patch the top-level torch.load
torch.load = _load_with_weights_false

import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from TTS.api import TTS


# Test phrases covering different scenarios
TEST_PHRASES = [
    # Short (tests response time)
    "Hello! How can I help you?",
    "Thank you for calling.",
    
    # Medium (typical responses)
    "Your ITR has been filed successfully. You will receive a confirmation email shortly.",
    "Let me check the status of your refund. Please wait a moment.",
    
    # Numbers (tests pronunciation)
    "Your refund amount is Rs. 15,750 and will be credited within 5 to 7 working days.",
    "Please enter your 12 digit Aadhaar number followed by the hash key.",
    
    # Hindi-English mix
    "Namaste! Aapka swagat hai. How may I assist you today?",
    "Aapka refund process ho raha hai. Please wait.",
    
    # Emotional tone
    "I understand your concern. Let me help you resolve this issue.",
    "Great news! Your verification has been completed successfully.",
]


def load_model(device: str) -> TTS:
    """Load XTTS v2 model (same as tts_server.py)."""
    print(f"Loading XTTS v2 on {device}...")
    start = time.time()
    
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    
    print(f"Model loaded in {time.time() - start:.1f}s")
    return model


def synthesize(tts: TTS, text: str, speaker_wav: str, language: str, speed: float) -> tuple:
    """
    Synthesize speech (mirrors tts_server.py _synthesize_wav_bytes).
    Returns (wav_samples, inference_time).
    """
    start = time.time()
    
    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        speed=speed,
    )
    
    return wav, time.time() - start


def save_wav(tts: TTS, wav: list, path: str):
    """Save WAV file (same as tts_server.py)."""
    tts.synthesizer.save_wav(wav=wav, path=path)


def main():
    parser = argparse.ArgumentParser(description="TTS Voice Tester")
    parser.add_argument(
        "--reference",
        default="/home/developer/ai-calling/abhishek_reference.wav",
        help="Reference voice WAV file",
    )
    parser.add_argument(
        "--output",
        default="/home/developer/ai-calling/logs/tts_tests",
        help="Output directory",
    )
    parser.add_argument(
        "--phrases",
        type=int,
        nargs="+",
        help="Phrase numbers to test (1-indexed)",
    )
    parser.add_argument(
        "--custom-text",
        help="Custom text to synthesize",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (0.5-2.0)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "hi"],
    )
    parser.add_argument(
        "--compare-speeds",
        action="store_true",
        help="Compare speeds 0.8, 0.9, 1.0, 1.1, 1.2",
    )
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.reference):
        print(f"Error: Reference not found: {args.reference}")
        sys.exit(1)
    
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tts = load_model(device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select phrases
    if args.custom_text:
        tests = [(0, args.custom_text)]
    elif args.phrases:
        tests = [(i, TEST_PHRASES[i-1]) for i in args.phrases if i <= len(TEST_PHRASES)]
    else:
        tests = list(enumerate(TEST_PHRASES, 1))
    
    # Speed comparison
    if args.compare_speeds:
        print(f"\n{'='*50}")
        print("SPEED COMPARISON")
        print(f"{'='*50}")
        
        text = args.custom_text or TEST_PHRASES[0]
        print(f"Text: {text}")
        
        for spd in [0.8, 0.9, 1.0, 1.1, 1.2]:
            fname = f"speed_{spd}_{ts}.wav"
            print(f"\nSpeed {spd}x... ", end="")
            
            wav, elapsed = synthesize(tts, text, args.reference, args.language, spd)
            save_wav(tts, wav, str(outdir / fname))
            
            dur = len(wav) / 24000
            print(f"{elapsed:.2f}s (audio: {dur:.2f}s, RTF: {elapsed/dur:.2f}x)")
        
        print(f"\nSaved to: {outdir}")
        return
    
    # Normal mode
    print(f"\n{'='*50}")
    print("TTS VOICE TEST")
    print(f"Reference: {args.reference}")
    print(f"Output: {outdir}")
    print(f"{'='*50}\n")
    
    results = []
    
    for idx, text in tests:
        safe = text[:25].replace(" ", "_").replace(",", "")
        fname = f"phrase_{idx}_{safe}_{ts}.wav"
        
        print(f"#{idx}: {text[:50]}...")
        
        try:
            wav, elapsed = synthesize(tts, text, args.reference, args.language, args.speed)
            save_wav(tts, wav, str(outdir / fname))
            
            dur = len(wav) / 24000
            results.append((fname, elapsed, dur))
            
            print(f"  → {fname} ({elapsed:.2f}s, RTF: {elapsed/dur:.2f}x)\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Generated: {len(results)} files")
    if results:
        avg_rtf = sum(r[2]/r[1] for r in results) / len(results)
        print(f"Avg RTF: {avg_rtf:.2f}x")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
