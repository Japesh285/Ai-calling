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
    generate_tts,
)
from app.config.settings import (
    FREESWITCH_AUDIO_DUMP_DIR,
    FREESWITCH_RAW_DUMP_DIR,
    PCM_CHANNELS,
    PCM_FRAME_BYTES,
    PCMA_FRAME_BYTES,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    TTS_WORKERS,
    TTS_PLAYBACK_GUARD_MS,
)
from app.pipeline.speech_pipeline import (
    SpeechSegmenter,
    append_brain_response,
    process_transcript_to_reply_streaming,
    transcribe_audio,
    transcribe_audio_streaming,
)
from app.sessions.call_session import CallSession
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
PLAYBACK_LOCK = asyncio.Lock()
TTS_WORKER_COUNT = 1


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


def play_to_caller(uuid: str, pcm16_audio: bytes) -> None:
    if not uuid:
        raise ValueError("Call UUID is required for playback")
    if not pcm16_audio:
        raise ValueError("PCM16 audio is required for playback")

    def delayed_delete(path: str, delay: float) -> None:
        time.sleep(delay)
        try:
            os.unlink(path)
        except Exception:
            pass

    temp_path: str | None = None
    try:
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

        # Keep file alive long enough for FreeSWITCH to play it.
        # All chunks are broadcast nearly simultaneously but play sequentially —
        # a later chunk's file must survive while all earlier chunks play.
        # Use 3× audio duration (min 30s) as a safe margin.
        audio_seconds = len(pcm16_audio) / (8000 * 2)
        delete_delay = max(30.0, audio_seconds * 3)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path, delete_delay), daemon=True).start()
    except subprocess.TimeoutExpired:
        logger.error("Playback command timed out for call_uuid=%s", uuid)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path, 30.0), daemon=True).start()
    except Exception:
        logger.exception("Playback failed for call_uuid=%s", uuid)
        if temp_path:
            threading.Thread(target=delayed_delete, args=(temp_path, 30.0), daemon=True).start()


def stop_ai_playback(call_uuid: str) -> bool:
    """
    Stop AI playback by breaking current audio playback.
    Uses uuid_break with 'both' direction to stop playback without hanging up.
    Returns True if successful, False otherwise.
    """
    logger.info("Stopping AI playback for call_uuid=%s", call_uuid)
    # Do NOT use "both" — it breaks all queued apps including sleep(300000)
    # which would cause the dialplan to end and hang up the call.
    # Plain uuid_break only stops the current playback (depth=1 uuid_broadcast).
    command = [
        "/usr/local/freeswitch/bin/fs_cli",
        "-x",
        f"uuid_break {call_uuid}",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            logger.info("FreeSWITCH break stdout: %s", stdout)
        if stderr:
            logger.debug("FreeSWITCH break stderr: %s", stderr)
        if result.returncode != 0:
            logger.warning(
                "Playback break failed for call_uuid=%s code=%s",
                call_uuid,
                result.returncode,
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Playback break command timed out for call_uuid=%s", call_uuid)
        return False
    except Exception:
        logger.exception("Playback break failed for call_uuid=%s", call_uuid)
        return False


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
    response_tasks: set[asyncio.Task[None]] = set()
    utterance_states: dict[int, dict[str, object]] = {}
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None] = asyncio.Queue()
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None] = asyncio.Queue()
    playback_state = {"suppress_until": 0.0}
    ai_state = {
        "speaking": False,
        "speaking_until": 0.0,
        # Tracks when the raw audio finishes playing (no guard buffer).
        # uuid_break is ONLY safe to call before this time — after it, sleep(300000)
        # becomes the current FreeSWITCH application and uuid_break would kill the call.
        "playback_ends_at": 0.0,
        "generation": 0,
        "playback_token": 0,
        "sentence_seq": 0,
    }
    tts_tasks = [
        asyncio.create_task(
            _tts_generation_worker(
                sentence_queue,
                playback_queue,
                client_label,
                ai_state,
            )
        )
        for _ in range(TTS_WORKER_COUNT)
    ]
    playback_task = asyncio.create_task(
        _stream_playback_worker(
            playback_queue,
            client_label,
            call_uuid,
            playback_state,
            ai_state,
        )
    )

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

                if time.monotonic() < playback_state["suppress_until"] and not _is_ai_speaking(ai_state):
                    segmenter.reset()
                    continue

                segment_result = segmenter.process_frame(frame)

                # Handle barge-in detection
                if segment_result.speech_started and _is_ai_speaking(ai_state):
                    logger.info("Barge-in detected")
                    # Schedule barge-in handling without blocking the main loop
                    asyncio.create_task(
                        _handle_barge_in(
                            call_uuid,
                            sentence_queue,
                            playback_queue,
                            playback_state,
                            ai_state,
                            response_tasks,
                            segmenter,
                        )
                    )
                    # Continue processing this frame as part of new speech
                    # Don't skip it - let it be processed for the new utterance

                # Handle final segment completion
                if segment_result.segment_completed and segment_result.segment_audio:
                    state = utterance_states.setdefault(
                        segment_result.utterance_id,
                        {"llm_started": False},
                    )
                    if state.get("llm_started"):
                        utterance_states.pop(segment_result.utterance_id, None)
                        continue

                    task = asyncio.create_task(
                        _process_final_segment(
                            segment_result.segment_audio,
                            client_label,
                            segment_result.utterance_id,
                            sentence_queue,
                            ai_state,
                            state,
                        )
                    )
                    segment_tasks.add(task)
                    task.add_done_callback(segment_tasks.discard)
                    utterance_states.pop(segment_result.utterance_id, None)

    except WebSocketDisconnect:
        pass
    finally:
        trailing_segment = segmenter.flush()
        if trailing_segment.segment_completed and trailing_segment.segment_audio:
            state = utterance_states.setdefault(
                trailing_segment.utterance_id,
                {"llm_started": False},
            )
            if not state.get("llm_started"):
                task = asyncio.create_task(
                        _process_final_segment(
                        trailing_segment.segment_audio,
                        client_label,
                        trailing_segment.utterance_id,
                        sentence_queue,
                        ai_state,
                        state,
                    )
                )
                segment_tasks.add(task)
                task.add_done_callback(segment_tasks.discard)

        for response_task in list(response_tasks):
            if not response_task.done():
                response_task.cancel()

        if response_tasks:
            await asyncio.gather(*response_tasks, return_exceptions=True)

        if segment_tasks:
            await asyncio.gather(*segment_tasks, return_exceptions=True)

        await sentence_queue.put(None)
        for _ in range(TTS_WORKER_COUNT - 1):
            await sentence_queue.put(None)
        await asyncio.gather(*tts_tasks, return_exceptions=True)
        await playback_queue.put(None)
        await playback_task

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


async def _handle_barge_in(
    call_uuid: str,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None],
    playback_state: dict[str, float],
    ai_state: dict[str, int | float | bool],
    response_tasks: set[asyncio.Task[None]],
    segmenter: SpeechSegmenter,
) -> None:
    """
    Handle barge-in by stopping AI playback and clearing queues.
    This function is fire-and-forget and should not block the main loop.
    """
    try:
        old_generation = int(ai_state["generation"])
        ai_state["generation"] = old_generation + 1
        ai_state["speaking"] = False
        ai_state["speaking_until"] = 0.0
        ai_state["playback_token"] = int(ai_state["playback_token"]) + 1
        ai_state["sentence_seq"] = 0
        
        # Clear suppression immediately so user speech after barge-in flows through.
        # The old suppress_until was set during AI playback and could be 600ms+
        # in the future. Keeping it would silently reset the segmenter on every
        # frame for that window, erasing the user's utterance and producing no response.
        playback_state["suppress_until"] = 0.0
        playback_state["pure_audio_end"] = 0.0

        _clear_sentence_queue(sentence_queue)
        _clear_playback_queue(playback_queue)

        # Do NOT send uuid_break on barge-in.
        #
        # uuid_break during a uuid_broadcast creates a race condition in FreeSWITCH:
        # breaking a depth=1 broadcast can signal the depth=0 sleep(300000) to wake,
        # advancing the dialplan past its last instruction and hanging up the call.
        #
        # Python's playback queue is already cleared above — no new chunks will be
        # broadcast.  The chunk currently playing in FreeSWITCH will finish naturally
        # (at most 1-2 s) and FreeSWITCH will return to sleep(300000) on its own.
        # This is safe and avoids the drop entirely.

        for response_task in list(response_tasks):
            if not response_task.done():
                response_task.cancel()
        response_tasks.clear()

        # Do NOT reset the segmenter here.
        # The speech frames that triggered barge-in are already in the segmenter buffer.
        # Resetting would erase the user's utterance, so segment_completed never fires
        # and the system never responds. Let the buffer accumulate naturally.
        logger.info("Barge-in handled successfully, listening resumed")
    except Exception:
        logger.exception("Barge-in handling failed")


async def _process_streaming_segment(
    audio_bytes: bytes,
    client_label: str,
    utterance_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
    state: dict[str, object],
) -> None:
    """
    Process streaming STT audio and start LLM response incrementally.
    This allows the AI to start responding before the user finishes speaking.
    """
    try:
        if state.get("llm_started"):
            return

        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("Streaming STT returned empty transcript for %s", client_label)
            return
        if state.get("llm_started"):
            return

        state["llm_started"] = True
        generation = int(ai_state["generation"])
        response = await _stream_response_from_transcript(
            transcript,
            generation,
            sentence_queue,
            ai_state,
        )
        await append_brain_response(transcript, response)
        print(f"User (streaming): {transcript}")
        print(f"AI: {response}")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Streaming STT/LLM processing failed for %s utterance=%s",
            client_label,
            utterance_id,
        )


async def _start_response_from_partial_stt(
    audio_bytes: bytes,
    client_label: str,
    utterance_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
    state: dict[str, object],
) -> None:
    try:
        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("Partial STT returned empty transcript for %s", client_label)
            return
        if state.get("llm_started"):
            return

        state["llm_started"] = True
        generation = int(ai_state["generation"])
        response = await _stream_response_from_transcript(
            transcript,
            generation,
            sentence_queue,
            ai_state,
        )
        await append_brain_response(transcript, response)
        print(f"User: {transcript}")
        print(f"AI: {response}")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Partial STT/LLM processing failed for %s utterance=%s",
            client_label,
            utterance_id,
        )


async def _process_final_segment(
    audio_bytes: bytes,
    client_label: str,
    utterance_id: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
    state: dict[str, object],
) -> None:
    try:
        if state.get("llm_started"):
            return

        transcript = await transcribe_audio(audio_bytes)
        if not transcript:
            logger.info("STT returned empty transcript for %s", client_label)
            return
        if state.get("llm_started"):
            return

        state["llm_started"] = True
        generation = int(ai_state["generation"])
        response = await _stream_response_from_transcript(
            transcript,
            generation,
            sentence_queue,
            ai_state,
        )
        await append_brain_response(transcript, response)
        print(f"User: {transcript}")
        print(f"AI: {response}")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Speech segment response failed for %s utterance=%s", client_label, utterance_id)


async def _stream_response_from_transcript(
    transcript: str,
    generation: int,
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    ai_state: dict[str, int | float | bool],
) -> str:
    response_queue: asyncio.Queue[str | None] = asyncio.Queue()
    forward_task = asyncio.create_task(
        _forward_sentence_chunks(response_queue, sentence_queue, generation, ai_state)
    )
    try:
        return await process_transcript_to_reply_streaming(transcript, response_queue)
    finally:
        await response_queue.put(None)
        await forward_task


async def _forward_sentence_chunks(
    source_queue: asyncio.Queue[str | None],
    target_queue: asyncio.Queue[tuple[int, int, str] | None],
    generation: int,
    ai_state: dict[str, int | float | bool],
) -> None:
    while True:
        sentence = await source_queue.get()
        if sentence is None:
            break
        if int(ai_state["generation"]) != generation:
            continue
        # Filter out too-short sentences before queuing for TTS
        if len(sentence.strip()) < 10:
            continue
        ai_state["sentence_seq"] = int(ai_state["sentence_seq"]) + 1
        await target_queue.put((generation, int(ai_state["sentence_seq"]), sentence))


def _clear_sentence_queue(sentence_queue: asyncio.Queue[tuple[int, int, str] | None]) -> None:
    while True:
        try:
            sentence_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    logger.info("Playback queue cleared")


def _clear_playback_queue(playback_queue: asyncio.Queue[tuple[int, int, bytes] | None]) -> None:
    while True:
        try:
            playback_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    logger.info("Playback queue cleared")


def _is_ai_speaking(ai_state: dict[str, int | float | bool]) -> bool:
    return bool(ai_state["speaking"]) or time.monotonic() < float(ai_state["speaking_until"])


async def _tts_generation_worker(
    sentence_queue: asyncio.Queue[tuple[int, int, str] | None],
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None],
    client_label: str,
    ai_state: dict[str, int | float | bool],
) -> None:
    while True:
        item = await sentence_queue.get()

        if item is None:
            break

        generation, sequence, sentence = item
        current_gen = int(ai_state["generation"])
        if generation != current_gen:
            continue
        # Filter out too-short sentences
        if len(sentence.strip()) < 10:
            continue

        pcm16_audio = await generate_tts(sentence, language=None)
        if int(ai_state["generation"]) != current_gen:
            continue
        if not pcm16_audio:
            logger.info("No TTS audio generated for %s", client_label)
            continue

        logger.info("TTS chunk generated, queuing PCM16 for playback")
        await playback_queue.put((current_gen, sequence, pcm16_audio))


async def _stream_playback_worker(
    playback_queue: asyncio.Queue[tuple[int, int, bytes] | None],
    client_label: str,
    call_uuid: str,
    playback_state: dict[str, float],
    ai_state: dict[str, int | float | bool],
) -> None:
    pending_audio: dict[int, bytes] = {}
    playback_generation = int(ai_state["generation"])
    next_sequence = 1

    while True:
        item = await playback_queue.get()
        if item is None:
            break

        generation, sequence, pcm16_audio = item
        current_generation = int(ai_state["generation"])
        if generation > playback_generation:
            playback_generation = generation
            pending_audio.clear()
            next_sequence = 1
        if generation != current_generation or generation != playback_generation:
            if generation > current_generation:
                pending_audio.clear()
                next_sequence = 1
            continue

        pending_audio[sequence] = pcm16_audio
        while next_sequence in pending_audio:
            current_audio = pending_audio.pop(next_sequence)
            gen_at_playback = int(ai_state["generation"])
            if generation != gen_at_playback:
                pending_audio.clear()
                break
            if not ai_state["speaking"] and gen_at_playback != generation:
                break

            playback_seconds = (len(current_audio) / PCM_FRAME_BYTES) * 0.02
            now = time.monotonic()
            audio_end = max(playback_state["suppress_until"], now) + playback_seconds
            speaking_until = audio_end + (TTS_PLAYBACK_GUARD_MS / 1000.0)
            playback_state["suppress_until"] = speaking_until
            ai_state["speaking"] = True
            ai_state["speaking_until"] = speaking_until

            # Track pure audio end WITHOUT guard accumulation.
            # suppress_until carries guard time from every preceding chunk, so using
            # it here would make playback_ends_at overestimate by N×guard_ms — causing
            # uuid_break to fire after actual audio ends and hit sleep(300000).
            pure_audio_end = playback_state.get("pure_audio_end", now)
            pure_chunk_end = max(pure_audio_end, now) + playback_seconds
            playback_state["pure_audio_end"] = pure_chunk_end
            ai_state["playback_ends_at"] = pure_chunk_end
            ai_state["playback_token"] = int(ai_state["playback_token"]) + 1
            playback_token = int(ai_state["playback_token"])

            async with PLAYBACK_LOCK:
                if int(ai_state["generation"]) != generation:
                    pending_audio.clear()
                    break
                await asyncio.to_thread(play_to_caller, call_uuid, current_audio)
            logger.info("Audio chunk sent to FreeSWITCH")
            asyncio.create_task(_mark_playback_complete(ai_state, speaking_until, playback_token))
            next_sequence += 1


async def _mark_playback_complete(
    ai_state: dict[str, int | float | bool],
    speaking_until: float,
    playback_token: int,
) -> None:
    delay_seconds = max(0.0, speaking_until - time.monotonic())
    await asyncio.sleep(delay_seconds)
    if playback_token == int(ai_state["playback_token"]):
        ai_state["speaking"] = False
        ai_state["playback_ends_at"] = 0.0


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
