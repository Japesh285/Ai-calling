import time


class CallSession:
    def __init__(self) -> None:
        self.stt_buffer = bytearray()
        self.raw_audio_buffer = bytearray()
        self.inbound_audio_buffer = bytearray()
        self.frame_counter = 0
        self.speech_chunk_counter = 0
        self.last_flush_ts = time.monotonic()

    def add_inbound_chunk(self, audio_chunk: bytes) -> None:
        self.inbound_audio_buffer.extend(audio_chunk)
        self.frame_counter += 1

    def add_speech_chunk(self, audio_chunk: bytes) -> None:
        self.stt_buffer.extend(audio_chunk)
        self.speech_chunk_counter += 1

    def add_raw_chunk(self, raw_chunk: bytes) -> None:
        self.raw_audio_buffer.extend(raw_chunk)

    def pop_buffer(self) -> bytes:
        if not self.stt_buffer:
            return b""

        payload = bytes(self.stt_buffer)
        self.stt_buffer.clear()
        self.last_flush_ts = time.monotonic()
        return payload

    def seconds_since_last_flush(self) -> float:
        return time.monotonic() - self.last_flush_ts

    def get_inbound_audio(self) -> bytes:
        return bytes(self.inbound_audio_buffer)

    def get_raw_audio(self) -> bytes:
        return bytes(self.raw_audio_buffer)
