"""Speaker naming: what to ask about, what to suggest, and applying the answers.

Kept free of any GUI code so it can be tested; the dialog lives in speaker_dialog.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .merge import Segment
from .store import read_json, write_json_atomic

CLIP_SECONDS = 6.0
_EMAIL = re.compile(r"<([^>]+)>$")


@dataclass
class SpeakerPrompt:
    speaker_id: str
    quotes: list[str]
    clip: tuple[float, float] | None  # seconds into the meeting, for "Play"
    seconds_spoken: float


def load_segments(folder: Path) -> list[Segment]:
    return [Segment.from_dict(d) for d in read_json(folder / "transcript.json")["segments"]]


def unnamed_speakers(folder: Path) -> list[SpeakerPrompt]:
    """Far-end speakers from diarization that have no name yet, most talkative first."""
    tpath = folder / "transcript.json"
    if not tpath.exists():
        return []
    meta = read_json(folder / "meta.json")
    names = meta.get("speaker_names", {})
    by_id: dict[str, list[Segment]] = {}
    for seg in load_segments(folder):
        if seg.channel == "others" and seg.label and seg.speaker not in names:
            by_id.setdefault(seg.speaker, []).append(seg)
    prompts = []
    for sid, segs in by_id.items():
        longest = sorted(segs, key=lambda s: len(s.text), reverse=True)
        quotes = [s.text for s in longest[:3] if s.text.strip()]
        best = max(segs, key=lambda s: s.end - s.start)
        clip = (best.start, min(best.end, best.start + CLIP_SECONDS))
        prompts.append(SpeakerPrompt(sid, quotes, clip, round(sum(s.end - s.start for s in segs), 1)))
    prompts.sort(key=lambda p: p.seconds_spoken, reverse=True)
    return prompts


def _email(person: str) -> str:
    m = _EMAIL.search(person.strip())
    return (m.group(1) if m else person).strip().lower()


def _display(person: str) -> str:
    return _EMAIL.sub("", person.strip()).strip() or person.strip()


def name_suggestions(root: Path, folder: Path, my_emails: list[str]) -> list[str]:
    """Invitees of this meeting first, then people named in earlier meetings, most frequent first."""
    meta = read_json(folder / "meta.json")
    mine = {e.lower() for e in my_emails}
    out: list[str] = []
    event = meta.get("calendar_event") or {}
    people = list(event.get("attendees", [])) + ([event["organizer"]] if event.get("organizer") else [])
    for p in people:
        if _email(p) not in mine and _display(p) not in out:
            out.append(_display(p))
    seen: Counter[str] = Counter()
    for meta_path in root.glob("*/meta.json"):
        if meta_path.parent == folder:
            continue
        try:
            other = read_json(meta_path)
        except Exception:
            continue
        seen.update(v for v in other.get("speaker_names_user", {}).values() if v)
    out += [name for name, _ in seen.most_common() if name not in out]
    return out


def apply_names(folder: Path, changes: dict[str, str]) -> dict[str, str]:
    """Set (or with an empty name, clear) display names, then rewrite the transcript.

    User-set names are kept apart from automatic ones so re-transcription keeps them.
    Giving two speaker ids the same name merges them in the transcript.
    """
    from .transcribe import write_transcript

    meta = read_json(folder / "meta.json")
    segments = load_segments(folder)
    ids = {s.speaker for s in segments}
    unknown = sorted(set(changes) - ids)
    if unknown:
        raise KeyError(f"Unknown speaker(s) {', '.join(unknown)}. Speakers in this meeting: {', '.join(sorted(ids))}")
    user_set = dict(meta.get("speaker_names_user", {}))
    names = dict(meta.get("speaker_names", {}))
    for sid, name in changes.items():
        name = name.strip()
        if name:
            user_set[sid] = names[sid] = name
        else:
            user_set.pop(sid, None)
            names.pop(sid, None)
    meta["speaker_names_user"] = user_set
    meta["speaker_names"] = names
    write_transcript(folder, meta, segments, meta.get("warnings", []))
    write_json_atomic(folder / "meta.json", meta)
    return names


def mark_prompted(folder: Path, outcome: str) -> None:
    meta = read_json(folder / "meta.json")
    meta["names_prompt"] = outcome  # "saved", "later" or "dismissed"
    write_json_atomic(folder / "meta.json", meta)


def should_prompt(folder: Path) -> bool:
    try:
        meta = read_json(folder / "meta.json")
    except Exception:
        return False
    return meta.get("names_prompt") != "dismissed" and bool(unnamed_speakers(folder))


def load_clip(audio_dir: Path, start: float, end: float, channel: str = "others") -> tuple[np.ndarray, int] | None:
    """Read just [start, end) seconds of a channel across its chunk files, without loading the meeting."""
    import soundfile as sf

    chunks = sorted(audio_dir.glob(f"{channel}_*.flac"))
    if not chunks or end <= start:
        return None
    parts, offset, rate = [], 0, None
    for path in chunks:
        info = sf.info(path)
        rate = rate or info.samplerate
        a, b = int(start * rate), int(end * rate)
        lo, hi = max(a - offset, 0), min(b - offset, info.frames)
        if lo < hi:
            data, _ = sf.read(path, start=lo, stop=hi, dtype="float32", always_2d=False)
            parts.append(data)
        offset += info.frames
        if offset >= b:
            break
    if not parts:
        return None
    return np.concatenate(parts), rate


def launch_dialog(root: Path, folder: Path) -> subprocess.Popen | None:
    """Open the naming dialog in its own process so it never blocks recording or the tray."""
    if not should_prompt(folder):
        return None
    return subprocess.Popen([sys.executable, "-m", "transcript_assistant.cli", "--root", str(root),
                             "name-speakers", str(folder)])
