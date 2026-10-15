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
    fallback: object = None  # device to try if `device` can't be opened (same .recorder() interface)


def _com_init() -> None:
    # Each capture thread needs COM initialised on Windows.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0)


class PortAudioMic:
    """A microphone through PortAudio (sounddevice).

    soundcard asserts that a device's shared-mode format is 32-bit float
    WAVEFORMATEXTENSIBLE and raises a bare AssertionError for mics that report
    anything else. PortAudio converts whatever the device delivers.
    """

    def __init__(self, device: str = ""):
        self.device = device or None  # name fragment; None means the Windows default
        self.name = f"{device or 'default microphone'} (PortAudio)"

    def recorder(self, samplerate: int, blocksize: int):
        return _PortAudioRecorder(samplerate, blocksize, self.device)


class _PortAudioRecorder:
    def __init__(self, samplerate: int, blocksize: int, device: str | None = None):
        self.samplerate, self.blocksize, self.device = samplerate, blocksize, device
        self.stream = None

    def __enter__(self):
        import sounddevice as sd

        self.stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.blocksize, device=self.device,
                                     channels=1, dtype="float32")
        self.stream.start()
        return self

    def record(self, numframes: int):
        data, _overflowed = self.stream.read(numframes)  # dropped input is padded by frames_to_pad
        return data

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
        return False


def _open(ch: _Channel, block: int):
    try:
        ctx = ch.device.recorder(samplerate=CAPTURE_RATE, blocksize=block)
        return ctx, ctx.__enter__()
    except Exception as exc:
        if ch.fallback is None:
            raise
        first = f"{type(exc).__name__}: {exc}".rstrip(": ")
        ctx = ch.fallback.recorder(samplerate=CAPTURE_RATE, blocksize=block)
        rec = ctx.__enter__()
        ch.stats.device = f"{getattr(ch.fallback, 'name', ch.fallback)} (fallback; {ch.stats.device} failed: {first})"
        return ctx, rec


def _capture(ch: _Channel, t0: float, stop: threading.Event) -> None:
    _com_init()
    block = int(CAPTURE_RATE * BLOCK_SECONDS)
    try:
        ctx, rec = _open(ch, block)
        try:
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
        finally:
            ctx.__exit__(None, None, None)
    except Exception as exc:  # surfaced in meta.json, never swallowed
        ch.stats.error = f"{type(exc).__name__}: {exc}".rstrip(": ")
    finally:
        ch.writer.close()


def _zeros(n: int):
    import numpy as np

    return np.zeros(n, dtype=np.float32)


def pick_devices(speaker_device: str = "", mic_device: str = ""):
    """(output to loop back, microphone): the Windows defaults unless a name fragment is configured."""
    import soundcard as sc

    speaker = _find(sc.all_speakers(), speaker_device, "speaker_device") if speaker_device else sc.default_speaker()
    mic = _find(sc.all_microphones(), mic_device, "mic_device") if mic_device else sc.default_microphone()
    return speaker, mic


def _find(devices, fragment: str, setting: str):
    hits = [d for d in devices if fragment.lower() in d.name.lower()]
    if len(hits) != 1:
        names = ", ".join(d.name for d in (hits or devices))
        raise ValueError(f"{setting} {fragment!r} matches {len(hits)} devices; pick one of: {names}")
    return hits[0]


@dataclass
class Recording:
    audio_dir: Path
    on_others_audio: object = None  # callable(np.ndarray) fed every far-end block, e.g. live naming
    speaker_device: str = ""
    mic_device: str = ""
    started_monotonic: float = 0.0
    channels: list[_Channel] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        import soundcard as sc

        speaker, mic = pick_devices(self.speaker_device, self.mic_device)
        devices = {
            "others": sc.get_microphone(id=speaker.id, include_loopback=True),
            "mic": mic,
        }
        self.started_monotonic = time.monotonic()
        for name, dev in devices.items():
            ch = _Channel(name, dev, ChunkWriter(self.audio_dir, name),
                          ChannelStats(name, device=str(getattr(dev, "name", dev))),
                          listener=self.on_others_audio if name == "others" else None,
                          fallback=PortAudioMic(self.mic_device) if name == "mic" else None)
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

    def all_failed(self) -> bool:
        return bool(self.channels) and all(c.stats.error for c in self.channels)

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
