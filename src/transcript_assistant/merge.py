"""Combine the two channels' segments into one speaker-labeled transcript."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

SPEAKER_LABELS = {"mic": "Me", "others": "Others"}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    channel: str  # "mic" or "others"

    @property
    def speaker(self) -> str:
        return SPEAKER_LABELS.get(self.channel, self.channel)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["speaker"] = self.speaker
        return d


def _overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _is_echo(m: Segment, o: Segment, min_ratio: float) -> bool:
    if _overlap(m, o) <= 0:
        return False
    # Short replies ("yeah", "okay") collide by chance, so demand a near-exact match.
    threshold = 0.85 if len(m.text.split()) < 4 else min_ratio
    return _similar(m.text, o.text) >= threshold


def drop_echo(mic: list[Segment], others: list[Segment], min_ratio: float = 0.6) -> tuple[list[Segment], int]:
    """Remove mic segments that are just the speakers bleeding into the microphone.

    Without a headset the mic hears the other participants too, which would
    duplicate their words under "Me". A mic segment is echo when it overlaps an
    "others" segment in time and says nearly the same thing.
    """
    kept, dropped = [], 0
    for m in mic:
        echo = any(_is_echo(m, o, min_ratio) for o in others)
        if echo:
            dropped += 1
        else:
            kept.append(m)
    return kept, dropped


def merge(mic: list[Segment], others: list[Segment]) -> tuple[list[Segment], int]:
    mic, dropped = drop_echo(mic, others)
    combined = sorted(mic + others, key=lambda s: (s.start, s.end))
    return combined, dropped


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_markdown(title: str, started_at: str, segments: list[Segment], warnings: list[str]) -> str:
    lines = [f"# {title}", "", f"Started: {started_at}", ""]
    if warnings:
        lines += ["> **Capture warnings**", ">"] + [f"> - {w}" for w in warnings] + [""]
    prev = None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker != prev:
            lines += ["", f"**{seg.speaker}** [{fmt_ts(seg.start)}]"]
            prev = seg.speaker
        lines.append(text)
    return "\n".join(lines).strip() + "\n"
