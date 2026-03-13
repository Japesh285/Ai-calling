from __future__ import annotations

import audioop
from collections import Counter
from math import log2


PCM16_SAMPLE_WIDTH = 2
HEX_PREVIEW_BYTES = 16
PCM16_NEAR_SILENCE_RMS = 100


def inspect_packet(packet: bytes) -> dict[str, int | float | str | bool]:
    packet_size = len(packet)
    first_bytes = packet[:HEX_PREVIEW_BYTES].hex()
    unique_values = len(set(packet))
    entropy = _byte_entropy(packet)
    pcm16_rms = 0
    pcm16_interpret_ok = False

    if packet_size >= PCM16_SAMPLE_WIDTH and packet_size % PCM16_SAMPLE_WIDTH == 0:
        try:
            pcm16_rms = audioop.rms(packet, PCM16_SAMPLE_WIDTH)
            pcm16_interpret_ok = True
        except audioop.error:
            pcm16_interpret_ok = False

    likely = _classify_packet(
        packet_size=packet_size,
        pcm16_rms=pcm16_rms,
        entropy=entropy,
    )

    return {
        "packet_bytes": packet_size,
        "first_bytes": first_bytes,
        "unique_bytes": unique_values,
        "entropy": round(entropy, 3),
        "pcm16_interpret_ok": pcm16_interpret_ok,
        "pcm16_rms": pcm16_rms,
        "likely": likely,
    }


def _classify_packet(
    *,
    packet_size: int,
    pcm16_rms: int,
    entropy: float,
) -> str:
    if packet_size == 0:
        return "empty"
    if pcm16_rms <= PCM16_NEAR_SILENCE_RMS:
        return "pcm16_silence"
    if entropy < 1.0:
        return "low_entropy"
    if pcm16_rms > PCM16_NEAR_SILENCE_RMS:
        return "pcm16_like"
    return "unknown"


def _byte_entropy(packet: bytes) -> float:
    if not packet:
        return 0.0

    counts = Counter(packet)
    total = len(packet)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return entropy
