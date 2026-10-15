"""Audio file helpers that do not touch any sound device (so they are testable anywhere)."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

CAPTURE_RATE = 48_000
WHISPER_RATE = 16_000
CHUNK_SECONDS = 300  # a crash loses at most the open chunk's unflushed tail


class ChunkWriter:
    """Writes one mono channel as numbered FLAC chunks: <prefix>_0000.flac, ..."""

    def __init__(self, folder: Path, prefix: str, samplerate: int = CAPTURE_RATE,
                 chunk_seconds: int = CHUNK_SECONDS):
        self.folder = folder
        self.prefix = prefix
        self.samplerate = samplerate
        self.chunk_frames = chunk_seconds * samplerate
        self.frames_written = 0
        self._index = 0
        self._in_chunk = 0
        self._file: sf.SoundFile | None = None
        self.sum_sq = 0.0
        self.peak = 0.0

    def _open(self) -> None:
        path = self.folder / f"{self.prefix}_{self._index:04d}.flac"
        self._file = sf.SoundFile(path, mode="w", samplerate=self.samplerate,
                                  channels=1, format="FLAC", subtype="PCM_16")
        self._in_chunk = 0

    def write(self, mono: np.ndarray) -> None:
        mono = np.clip(np.asarray(mono, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if mono.size:
            self.sum_sq += float(np.dot(mono, mono))
            self.peak = max(self.peak, float(np.max(np.abs(mono))))
        pos = 0
        while pos < mono.size:
            if self._file is None:
                self._open()
            take = min(mono.size - pos, self.chunk_frames - self._in_chunk)
            self._file.write(mono[pos:pos + take])
            self._in_chunk += take
            self.frames_written += take
            pos += take
            if self._in_chunk >= self.chunk_frames:
                self._file.close()
                self._file = None
                self._index += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def rms(self) -> float:
        return (self.sum_sq / self.frames_written) ** 0.5 if self.frames_written else 0.0


def to_mono(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    return block.mean(axis=1) if block.ndim == 2 else block


def frames_to_pad(frames_written: int, block_frames: int, elapsed_s: float,
                  samplerate: int = CAPTURE_RATE, tolerance_s: float = 0.25) -> int:
    """Silence to insert before a block so the channel stays on the wall clock.

    Windows loopback delivers no packets while nothing is playing, so without
    padding the 'others' track would shrink and drift out of sync with the mic.
    The block that just arrived is assumed to end now.
    """
    expected = int(elapsed_s * samplerate)
    gap = expected - (frames_written + block_frames)
    return gap if gap > int(tolerance_s * samplerate) else 0


def load_channel(audio_dir: Path, prefix: str, target_rate: int = WHISPER_RATE) -> np.ndarray:
    """Concatenate a channel's chunks and resample to the rate Whisper expects."""
    parts = []
    rate = None
    for path in sorted(audio_dir.glob(f"{prefix}_*.flac")):
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if rate is not None and sr != rate:
            raise ValueError(f"{path.name}: sample rate {sr} differs from earlier chunks ({rate})")
        rate = sr
        parts.append(to_mono(data))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(parts)
    if rate != target_rate:
        from scipy.signal import resample_poly

        g = gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // g, rate // g).astype(np.float32)
    return audio
