"""
Test script for Indic TTS streaming endpoint.
Saves streamed audio to file and measures latency.
"""
import httpx
import asyncio
import time

TTS_URL = "http://localhost:8001"  # Change to your Indic TTS server port

async def test_streaming():
    print("="*60)
    print("INDIC TTS STREAMING TEST")
    print("="*60)
    
    text = "नमस्ते आपका स्वागत है"
    print(f"Text: {text}")
    print(f"Endpoint: {TTS_URL}/synthesize/stream")
    print()
    
    output_path = "output_streaming_test.wav"
    first_chunk_time = None
    total_time = None
    bytes_received = 0
    
    start_time = time.monotonic()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{TTS_URL}/synthesize/stream",
            json={"text": text, "language": "hi"}
        ) as response:
            print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: {response.text}")
                return
            
            with open(output_path, 'wb') as f:
                async for chunk in response.aiter_bytes():
                    elapsed = time.monotonic() - start_time
                    bytes_received += len(chunk)
                    
                    if first_chunk_time is None:
                        first_chunk_time = elapsed
                        print(f"✓ First chunk received at {elapsed:.2f}s ({len(chunk)} bytes)")
                    
                    f.write(chunk)
            
            total_time = time.monotonic() - start_time
    
    print(f"✓ Last chunk received at {total_time:.2f}s")
    print(f"✓ Total bytes: {bytes_received:,}")
    print(f"✓ Output saved to: {output_path}")
    print()
    
    # Calculate stats
    print("="*60)
    print("LATENCY METRICS")
    print("="*60)
    print(f"Time to first audio: {first_chunk_time:.2f}s")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Improvement vs blocking: {(1 - first_chunk_time/total_time) * 100:.1f}% faster first audio")
    print()
    
    # Verify file
    import wave
    try:
        with wave.open(output_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
            print(f"✓ Valid WAV file: {duration:.2f}s duration")
    except Exception as e:
        print(f"✗ WAV file validation failed: {e}")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_streaming())
