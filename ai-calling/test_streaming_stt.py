"""
Test script for streaming STT functionality.
Tests the incremental transcription endpoint.
"""
import asyncio
import wave
import httpx


async def test_streaming_stt(audio_file_path: str, stt_server_url: str = "http://localhost:8000"):
    """
    Test streaming STT by sending audio chunks incrementally.
    """
    print(f"Testing streaming STT with file: {audio_file_path}")
    print(f"STT Server URL: {stt_server_url}/transcribe/stream")
    
    # Read the WAV file
    with wave.open(audio_file_path, "rb") as wav_file:
        # Verify audio format
        assert wav_file.getnchannels() == 1, "Audio must be mono"
        assert wav_file.getsampwidth() == 2, "Audio must be 16-bit PCM"
        assert wav_file.getframerate() == 8000, "Audio must be 8kHz"
        
        frame_size = 320  # 20ms at 8kHz, 16-bit, mono
        all_frames = wav_file.readframes(wav_file.getnframes())
    
    print(f"Total audio bytes: {len(all_frames)}")
    print(f"Frame size: {frame_size} bytes")
    print(f"Total frames: {len(all_frames) // frame_size}")
    
    # Stream audio chunks to the server
    async def generate_chunks():
        for i in range(0, len(all_frames), frame_size):
            chunk = all_frames[i:i + frame_size]
            yield chunk
            await asyncio.sleep(0.02)  # Simulate realtime streaming (20ms per frame)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{stt_server_url}/transcribe/stream",
                content=generate_chunks(),
                headers={"Content-Type": "audio/wav"},
            ) as response:
                response.raise_for_status()
                print("\n=== Streaming Transcription Results ===")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            print("\n[Streaming complete]")
                            break
                        if data.startswith("error: "):
                            print(f"\n[Error: {data[7:]}]")
                            continue
                        if data.strip():
                            print(f"Transcript: {data}")
    except httpx.ConnectError as exc:
        print(f"\nFailed to connect to STT server: {exc}")
        print("Make sure the STT server is running on port 8000")
        return False
    except Exception as exc:
        print(f"\nTest failed: {exc}")
        return False
    
    return True


async def test_batch_stt(audio_file_path: str, stt_server_url: str = "http://localhost:8000"):
    """
    Test batch STT by sending the entire audio file at once.
    """
    print(f"\n=== Testing Batch STT ===")
    print(f"STT Server URL: {stt_server_url}/transcribe")
    
    with wave.open(audio_file_path, "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 8000
        
        all_frames = wav_file.readframes(wav_file.getnframes())
    
    # Convert to WAV format for the API
    import io
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(8000)
        wav_out.writeframes(all_frames)
    
    wav_bytes = buffer.getvalue()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"audio": ("test.wav", wav_bytes, "audio/wav")}
            response = await client.post(f"{stt_server_url}/transcribe", files=files)
            response.raise_for_status()
            result = response.json()
            print(f"Transcript: {result.get('text', 'N/A')}")
            print(f"Language: {result.get('language', 'N/A')}")
    except httpx.ConnectError as exc:
        print(f"Failed to connect to STT server: {exc}")
        return False
    except Exception as exc:
        print(f"Test failed: {exc}")
        return False
    
    return True


async def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_streaming_stt.py <audio_file.wav> [stt_server_url]")
        print("Example: python test_streaming_stt.py /path/to/audio.wav http://localhost:8000")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    print("=" * 60)
    print("Streaming STT Test Suite")
    print("=" * 60)
    
    # Test batch STT first
    batch_success = await test_batch_stt(audio_file, server_url)
    
    if batch_success:
        print("\n✓ Batch STT test passed")
    else:
        print("\n✗ Batch STT test failed")
    
    # Test streaming STT
    streaming_success = await test_streaming_stt(audio_file, server_url)
    
    if streaming_success:
        print("\n✓ Streaming STT test passed")
    else:
        print("\n✗ Streaming STT test failed")
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
