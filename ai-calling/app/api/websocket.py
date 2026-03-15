import asyncio
import base64
import audioop
import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import wave

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.audio.tts_streamer import (
    convert_pcm16_bytes_to_pcma,
    generate_tts,
)
from app.config.settings import (
    BASE_DIR,
    FREESWITCH_AUDIO_DUMP_DIR,
    FREESWITCH_RAW_DUMP_DIR,
    PCM_CHANNELS,
    PCMA_FRAME_BYTES,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    TTS_PLAYBACK_GUARD_MS,
)
from app.pipeline.speech_pipeline import SpeechSegmenter, process_speech_segment_to_reply
from app.sessions.call_session import CallSession
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

def _dump_inbound_audio(audio_bytes: bytes) -> Path | None:
    if not audio_bytes:
        return None

    FREESWITCH_AUDIO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = FREESWITCH_AUDIO_DUMP_DIR / f"call_{timestamp}.wav"

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(PCM_SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)

    return output_path


def _dump_raw_audio(audio_bytes: bytes) -> Path | None:
    if not audio_bytes:
        return None

    FREESWITCH_RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = FREESWITCH_RAW_DUMP_DIR / f"call_{timestamp}.alaw"
    output_path.write_bytes(audio_bytes)
    return output_path


def play_to_caller(uuid: str, pcma_audio: bytes) -> None:
    if not uuid:
        raise ValueError("Call UUID is required for playback")
    if not pcma_audio:
        raise ValueError("PCMA audio is required for playback")

    def delayed_delete(path: str) -> None:
        time.sleep(5)
        try:
            os.unlink(path)
        except Exception:
            pass

    temp_path: str | None = None
    try:
        pcm16_audio = audioop.alaw2lin(pcma_audio, 2)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(pcm16_audio)

        command = [
            "/usr/local/freeswitch/bin/fs_cli",
            "-x",
            f"uuid_broadcast {uuid} {temp_path} aleg",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            logger.info("FreeSWITCH command stdout: %s", stdout)
        if stderr:
            logger.info("FreeSWITCH command stderr: %s", stderr)

        if result.returncode == 0:
            logger.info("Playback succeeded for call_uuid=%s file=%s", uuid, temp_path)
        else:
            logger.error(
                "Playback failed for call_uuid=%s code=%s file=%s",
                uuid,
                result.returncode,
                temp_path,
            )

        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path,), daemon=True).start()
    except subprocess.TimeoutExpired:
        logger.error("Playback command timed out for call_uuid=%s", uuid)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path,), daemon=True).start()
    except Exception:
        logger.exception("Playback failed for call_uuid=%s", uuid)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path,), daemon=True).start()


@router.websocket("/audio-stream")
@router.websocket("/audio")
@router.websocket("/ws/audio")
async def audio_stream(ws: WebSocket) -> None:
    await ws.accept()
    call_uuid = ws.query_params.get("uuid")
    if not call_uuid:
        logger.error("Missing FreeSWITCH call UUID in websocket query params")
        await ws.close(code=1008, reason="Missing call UUID")
        return

    logger.info("Connected FreeSWITCH call UUID: %s", call_uuid)
    session = CallSession()
    segmenter = SpeechSegmenter()
    client = getattr(ws, "client", None)
    client_label = f"{client.host}:{client.port}" if client else "unknown"
    pcm_buffer = bytearray()
    segment_tasks: set[asyncio.Task[None]] = set()
    send_lock = asyncio.Lock()
    playback_state = {"suppress_until": 0.0}

    try:
        while True:
            message = await ws.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            audio_chunk = message.get("bytes")
            text_frame = message.get("text")
            if text_frame and not audio_chunk:
                audio_chunk = _extract_audio_from_text_frame(text_frame)

            if not audio_chunk:
                continue

            pcm_buffer.extend(audio_chunk)

            while len(pcm_buffer) >= 320:
                frame = bytes(pcm_buffer[:320])
                del pcm_buffer[:320]

                session.add_raw_chunk(frame)
                session.add_inbound_chunk(frame)

                if time.monotonic() < playback_state["suppress_until"]:
                    segmenter.reset()
                    continue

                segment_result = segmenter.process_frame(frame)

                if segment_result.segment_completed and segment_result.segment_audio:
                    task = asyncio.create_task(
                        _process_segment_and_respond(
                            segment_result.segment_audio,
                            send_lock,
                            client_label,
                            call_uuid,
                            playback_state,
                        )
                    )
                    segment_tasks.add(task)
                    task.add_done_callback(segment_tasks.discard)

    except WebSocketDisconnect:
        pass
    finally:
        trailing_segment = segmenter.flush()
        if trailing_segment:
            task = asyncio.create_task(
                _process_segment_and_respond(
                    trailing_segment,
                    send_lock,
                    client_label,
                    call_uuid,
                    playback_state,
                )
            )
            segment_tasks.add(task)
            task.add_done_callback(segment_tasks.discard)

        if segment_tasks:
            await asyncio.gather(*segment_tasks, return_exceptions=True)

        raw_audio_path = await _save_raw_audio_dump(session)
        if raw_audio_path:
            logger.info("Saved raw FreeSWITCH audio to %s", raw_audio_path)

        inbound_audio_path = await _save_inbound_audio_dump(session)
        if inbound_audio_path:
            logger.info("Saved inbound FreeSWITCH audio to %s", inbound_audio_path)

        decoded_audio = session.get_inbound_audio()
        if decoded_audio:
            total_rms = audioop.rms(decoded_audio, PCM_SAMPLE_WIDTH_BYTES)
            duration_seconds = len(decoded_audio) / (
                PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
            )
            logger.info(
                "Call audio summary: duration=%.2fs decoded_bytes=%s raw_bytes=%s rms=%s",
                duration_seconds,
                len(decoded_audio),
                len(session.get_raw_audio()),
                total_rms,
            )


def _extract_audio_from_text_frame(text_frame: str) -> bytes | None:
    try:
        payload = json.loads(text_frame)
    except json.JSONDecodeError:
        logger.info("FreeSWITCH text frame: %s", text_frame)
        return None

    if isinstance(payload, dict):
        event_type = payload.get("event") or payload.get("type")
        audio_base64 = payload.get("audio") or payload.get("data")
        if event_type:
            logger.info("FreeSWITCH event: %s", event_type)
        if isinstance(audio_base64, str):
            try:
                return base64.b64decode(audio_base64)
            except Exception:
                logger.warning("Invalid base64 audio payload in text frame")
                return None

    logger.info("FreeSWITCH text payload received")
    return None


async def _process_segment_and_respond(
    audio_bytes: bytes,
    send_lock: asyncio.Lock,
    client_label: str,
    call_uuid: str | None,
    playback_state: dict[str, float],
) -> None:
    try:
        transcript, response = await process_speech_segment_to_reply(audio_bytes)
        if not transcript:
            logger.info("STT returned empty transcript for %s", client_label)
            return

        print(f"User: {transcript}")
        print(f"AI: {response}")

        async with send_lock:
            pcm16_audio = await generate_tts(response)
            if not pcm16_audio:
                logger.info("No TTS audio generated for %s", client_label)
                return

            pcma_audio = await convert_pcm16_bytes_to_pcma(pcm16_audio)
            if not pcma_audio:
                logger.info("No PCMA audio generated for %s", client_label)
                return

            if not call_uuid:
                logger.error("Cannot play response for %s without call UUID", client_label)
                return

            playback_seconds = (len(pcma_audio) / PCMA_FRAME_BYTES) * 0.02
            playback_state["suppress_until"] = (
                time.monotonic() + playback_seconds + (TTS_PLAYBACK_GUARD_MS / 1000.0)
            )
            await asyncio.to_thread(play_to_caller, call_uuid, pcma_audio)
            logger.info("Queued TTS playback for client=%s bytes=%s", client_label, len(pcma_audio))
    except Exception:
        logger.exception("Speech segment response failed for %s", client_label)


async def _save_inbound_audio_dump(session: CallSession) -> Path | None:
    inbound_audio = session.get_inbound_audio()
    if not inbound_audio:
        return None

    return await asyncio.to_thread(_dump_inbound_audio, inbound_audio)


async def _save_raw_audio_dump(session: CallSession) -> Path | None:
    raw_audio = session.get_raw_audio()
    if not raw_audio:
        return None

    return await asyncio.to_thread(_dump_raw_audio, raw_audio)
