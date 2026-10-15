"""Capture system output (everyone else) and the microphone (you) as two channels.

Uses WASAPI loopback through the `soundcard` package, so it records whatever the
meeting app plays, whichever app that is.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio import CAPTURE_RATE, ChunkWriter, frames_to_pad, to_mono

BLOCK_SECONDS = 0.5


@dataclass
class ChannelStats:
    name: str
    device: str = ""
    frames: int = 0
    padded_frames: int = 0
    rms: float = 0.0
    peak: float = 0.0
    error: str | None = None


@dataclass
class _Channel:
    name: str
    device: object
    writer: ChunkWriter
    stats: ChannelStats
    thread: threading.Thread | None = None
    listener: object = None  # optional callable(np.ndarray) that sees every block written


def _com_init() -> None:
    # Each capture thread needs COM initialised on Windows.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0)


def _capture(ch: _Channel, t0: float, stop: threading.Event) -> None:
    _com_init()
    block = int(CAPTURE_RATE * BLOCK_SECONDS)
    try:
        with ch.device.recorder(samplerate=CAPTURE_RATE, blocksize=block) as rec:
            while not stop.is_set():
                data = to_mono(rec.record(numframes=block))
                pad = frames_to_pad(ch.writer.frames_written, len(data), time.monotonic() - t0)
                if pad:
                    silence = _zeros(pad)
                    ch.writer.write(silence)
                    ch.stats.padded_frames += pad
                    if ch.listener is not None:
                        ch.listener(silence)
                ch.writer.write(data)
                if ch.listener is not None:
                    ch.listener(data)
    except Exception as exc:  # surfaced in meta.json, never swallowed
        ch.stats.error = f"{type(exc).__name__}: {exc}"
    finally:
        ch.writer.close()


def _zeros(n: int):
    import numpy as np

    return np.zeros(n, dtype=np.float32)


@dataclass
class Recording:
    audio_dir: Path
    on_others_audio: object = None  # callable(np.ndarray) fed every far-end block, e.g. live naming
    started_monotonic: float = 0.0
    channels: list[_Channel] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        import soundcard as sc

        speaker = sc.default_speaker()
        devices = {
            "others": sc.get_microphone(id=str(speaker.name), include_loopback=True),
            "mic": sc.default_microphone(),
        }
        self.started_monotonic = time.monotonic()
        for name, dev in devices.items():
            ch = _Channel(name, dev, ChunkWriter(self.audio_dir, name),
                          ChannelStats(name, device=str(getattr(dev, "name", dev))),
                          listener=self.on_others_audio if name == "others" else None)
            ch.thread = threading.Thread(target=_capture, args=(ch, self.started_monotonic, self._stop),
                                         name=f"capture-{name}", daemon=True)
            self.channels.append(ch)
            ch.thread.start()

    def stop(self) -> tuple[float, list[ChannelStats]]:
        self._stop.set()
        for ch in self.channels:
            ch.thread.join(timeout=10)
            ch.writer.close()
            ch.stats.frames = ch.writer.frames_written
            ch.stats.rms = round(ch.writer.rms, 6)
            ch.stats.peak = round(ch.writer.peak, 6)
        return time.monotonic() - self.started_monotonic, [c.stats for c in self.channels]

    def any_failed(self) -> bool:
        return any(c.stats.error for c in self.channels)

    def failure_detail(self) -> str:
        return "; ".join(f"{c.name}: {c.stats.error}" for c in self.channels if c.stats.error)


def capture_warnings(duration_s: float, stats: list[ChannelStats]) -> list[str]:
    """Flag recordings that look wrong instead of letting them pass silently."""
    warnings = []
    for s in stats:
        captured = s.frames / CAPTURE_RATE
        if s.error:
            warnings.append(f"{s.name}: capture stopped with error: {s.error}")
        if duration_s > 30 and captured < 0.9 * duration_s:
            warnings.append(f"{s.name}: captured {captured:.0f}s of {duration_s:.0f}s")
        if s.frames and s.peak < 0.001:
            warnings.append(f"{s.name}: channel is silent (wrong device or muted?)")
        if s.name == "mic" and s.padded_frames > 2 * CAPTURE_RATE:
            warnings.append(f"mic: {s.padded_frames / CAPTURE_RATE:.1f}s of dropouts padded with silence")
    return warnings
