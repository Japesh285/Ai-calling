#!/usr/bin/env python3
"""
Standalone TTS Voice Testing Script

Test different voice synthesis outputs using your reference voice.
Compares multiple text samples and saves them for evaluation.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from TTS.tts.models.xtts import Xtts


# Safe unpickle for TTS models (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([
    XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig, Xtts
])


# Test phrases for voice quality evaluation
TEST_PHRASES = [
    # Short greetings
    "Hello! How can I help you today?",
    "Hi there! Welcome to FSK India.",
    
    # Professional responses
    "Your ITR has been successfully filed. You will receive a confirmation email shortly.",
    "Let me check the status of your refund. Please wait a moment.",
    
    # Numbers and dates (tests pronunciation)
    "Your refund amount is Rs. 15,750 and will be credited within 5 to 7 working days.",
    "The deadline for filing is July 31st, 2026.",
    
    # Hindi-English mix (common in Indian context)
    "Namaste! Aapka swagat hai. How may I assist you with your tax query?",
    "Aapka refund process ho raha hai. Your refund is being processed.",
    
    # Emotional variations
    "I understand your concern. Let me help you resolve this issue.",
    "Great news! Your verification has been completed successfully.",
    
    # Technical terms
    "Please verify your Aadhaar number and PAN card details before proceeding.",
    "Form 16 and Form 26AS are essential documents for ITR filing.",
]


def load_tts_model(device: str = "cuda") -> TTS:
    """Load XTTS v2 model."""
    print(f"Loading TTS model on {device}...")
    start = time.time()
    
    # PyTorch 2.6+ compatibility: monkey-patch torch.load to use weights_only=False
    import torch.serialization as serialization
    
    original_load = serialization.load
    
    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)
    
    serialization.load = patched_load
    
    try:
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    finally:
        serialization.load = original_load
    
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return model


def synthesize_voice(
    tts: TTS,
    text: str,
    speaker_wav: str,
    language: str = "en",
    speed: float = 1.0,
) -> tuple[list[float], float]:
    """
    Synthesize speech and return audio samples.
    
    Returns:
        Tuple of (audio_samples, inference_time_seconds)
    """
    start = time.time()
    
    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        speed=speed,
    )
    
    elapsed = time.time() - start
    return wav, elapsed


def save_wav_file(tts: TTS, wav: list[float], filepath: str) -> None:
    """Save audio samples to WAV file."""
    tts.synthesizer.save_wav(wav=wav, path=filepath)


def main():
    parser = argparse.ArgumentParser(description="TTS Voice Quality Tester")
    parser.add_argument(
        "--reference",
        type=str,
        default="/home/developer/ai-calling/abhishek_reference.wav",
        help="Path to reference voice WAV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/developer/ai-calling/logs/tts_tests",
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--phrases",
        type=int,
        nargs="+",
        default=None,
        help="Phrase indices to test (e.g., --phrases 1 3 5)",
    )
    parser.add_argument(
        "--custom-text",
        type=str,
        default=None,
        help="Custom text to synthesize",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (0.5=slow, 2.0=fast)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "hi"],
        help="Language code",
    )
    parser.add_argument(
        "--compare-speeds",
        action="store_true",
        help="Generate same phrase at different speeds (0.8, 1.0, 1.2)",
    )
    args = parser.parse_args()
    
    # Validate reference file
    if not os.path.exists(args.reference):
        print(f"Error: Reference file not found: {args.reference}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    tts = load_tts_model(device)
    
    # Generate timestamp for this test session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which phrases to test
    if args.custom_text:
        phrases_to_test = [(0, args.custom_text)]
    elif args.phrases:
        phrases_to_test = [(i, TEST_PHRASES[i-1]) for i in args.phrases if i <= len(TEST_PHRASES)]
    else:
        phrases_to_test = list(enumerate(TEST_PHRASES, 1))
    
    # Speed comparison mode
    if args.compare_speeds:
        print(f"\n{'='*60}")
        print("SPEED COMPARISON MODE")
        print(f"{'='*60}")
        
        test_text = args.custom_text or TEST_PHRASES[0]
        speeds = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        print(f"\nText: {test_text[:80]}...")
        print(f"\nGenerating at different speeds:")
        
        for speed in speeds:
            filename = f"speed_{speed:.1f}_{timestamp}.wav"
            filepath = output_dir / filename
            
            print(f"  Speed {speed}x... ", end="", flush=True)
            wav, elapsed = synthesize_voice(tts, test_text, args.reference, args.language, speed)
            save_wav_file(tts, wav, str(filepath))
            
            audio_duration = len(wav) / 24000  # XTTS outputs at 24kHz
            rtf = elapsed / audio_duration
            print(f"{elapsed:.2f}s (RTF: {rtf:.2f}x, audio: {audio_duration:.2f}s)")
        
        print(f"\nSaved to: {output_dir}")
        print("\nCompare the files to find optimal speed setting.")
        return
    
    # Normal testing mode
    print(f"\n{'='*60}")
    print(f"TTS VOICE TEST SESSION")
    print(f"Reference: {args.reference}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    results = []
    total_time = 0
    
    for idx, text in phrases_to_test:
        # Clean text for filename
        safe_text = text[:30].replace(" ", "_").replace(",", "").replace(".", "")
        filename = f"phrase_{idx:02d}_{safe_text}_{timestamp}.wav"
        filepath = output_dir / filename
        
        print(f"Phrase {idx}: {text[:60]}...")
        
        try:
            wav, elapsed = synthesize_voice(tts, text, args.reference, args.language, args.speed)
            save_wav_file(tts, wav, str(filepath))
            
            audio_duration = len(wav) / 24000  # 24kHz output
            rtf = elapsed / audio_duration
            total_time += elapsed
            
            results.append({
                "idx": idx,
                "text": text,
                "file": filename,
                "inference_time": elapsed,
                "audio_duration": audio_duration,
                "rtf": rtf,
            })
            
            print(f"  → {filename}")
            print(f"     Inference: {elapsed:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.2f}x\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated: {len(results)}/{len(phrases_to_test)} files")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average RTF: {sum(r['rtf'] for r in results)/len(results):.2f}x" if results else "")
    print(f"\nOutput directory: {output_dir}")
    
    # Print file list for easy playback
    print(f"\nFiles generated:")
    for r in results:
        print(f"  {r['file']}")


if __name__ == "__main__":
    main()
