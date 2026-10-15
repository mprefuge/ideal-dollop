"""Split the "Others" channel into individual speakers, and name them when it is safe to.

Diarization uses pyannote's speaker-diarization-3.1 pipeline on the GPU. It is
optional (`pip install -e .[diarize]`) and needs a free Hugging Face token with
the model's terms accepted, because the model is gated.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .audio import WHISPER_RATE
from .merge import Segment

PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
NEAREST_TURN_S = 1.0  # a word this close to a turn is assigned to it


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def load_pipeline(token: str | None = None, device: str = "cuda"):
    import torch
    from pyannote.audio import Pipeline

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Diarization needs a Hugging Face token in the HF_TOKEN environment variable.")
    try:
        pipeline = Pipeline.from_pretrained(PIPELINE_NAME, token=token)
    except TypeError:  # pyannote.audio 3.x
        pipeline = Pipeline.from_pretrained(PIPELINE_NAME, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(f"Could not load {PIPELINE_NAME}; accept its terms on huggingface.co first.")
    pipeline.to(torch.device(device))
    return pipeline


def run_pipeline(pipeline, audio) -> list[Turn]:
    import torch

    result = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": WHISPER_RATE})
    annotation = getattr(result, "speaker_diarization", result)  # pyannote 4.x wraps the annotation
    return [Turn(float(t.start), float(t.end), str(spk))
            for t, _, spk in annotation.itertracks(yield_label=True)]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_at(start: float, end: float, turns: list[Turn]) -> str | None:
    best, best_ov = None, 0.0
    for t in turns:
        ov = _overlap(start, end, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best is not None:
        return best
    mid = (start + end) / 2
    near = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)), default=None)
    if near and min(abs(mid - near.start), abs(mid - near.end)) <= NEAREST_TURN_S:
        return near.speaker
    return None


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """Label each segment with a speaker, splitting segments whose words change speaker.

    Raw pyannote labels (SPEAKER_00, ...) become "Speaker 1", "Speaker 2", ...
    in order of first appearance, so the numbering reads naturally.
    """
    if not turns:
        return segments
    pieces: list[tuple[str | None, Segment]] = []
    for seg in segments:
        if not seg.words:
            pieces.append((_speaker_at(seg.start, seg.end, turns), seg))
            continue
        run: list[tuple[float, float, str]] = []
        run_spk: str | None = None
        for w in seg.words:
            spk = _speaker_at(w[0], w[1], turns) or run_spk
            if run and spk != run_spk:
                pieces.append((run_spk, _from_words(run, seg.channel)))
                run = []
            run.append(w)
            run_spk = spk
        if run:
            pieces.append((run_spk, _from_words(run, seg.channel)))

    order: dict[str, str] = {}
    out = []
    for raw, seg in pieces:
        if raw is not None and raw not in order:
            order[raw] = f"Speaker {len(order) + 1}"
        seg.label = order.get(raw) if raw is not None else None
        out.append(seg)
    return out


def _from_words(words: list[tuple[float, float, str]], channel: str) -> Segment:
    text = "".join(w[2] for w in words).strip()
    return Segment(round(words[0][0], 2), round(words[-1][1], 2), text, channel, words=list(words))


_EMAIL = re.compile(r"<([^>]+)>$")


def _email(person: str) -> str:
    m = _EMAIL.search(person.strip())
    return (m.group(1) if m else person).strip().lower()


def _display(person: str) -> str:
    person = person.strip()
    return _EMAIL.sub("", person).strip() or person


def calendar_names(segments: list[Segment], event: dict | None, my_emails: list[str]) -> tuple[dict[str, str], str]:
    """Name the far-end speaker only when the calendar leaves no doubt.

    With exactly one other person invited and at most one far-end voice heard,
    that voice is that person. Anything less certain stays "Speaker N" for the
    user to rename, because a confident wrong name is worse than a number.
    """
    if not event:
        return {}, ""
    mine = {e.lower() for e in my_emails}
    people = list(event.get("attendees", []))
    if event.get("organizer"):
        people.append(event["organizer"])
    others = {}
    for p in people:
        email = _email(p)
        if email and email not in mine:
            others.setdefault(email, _display(p))
    if len(others) != 1:
        return {}, ""
    far_ids = {s.speaker for s in segments if s.channel == "others"}
    if len(far_ids) != 1:
        return {}, ""
    (name,) = others.values()
    return {far_ids.pop(): name}, "calendar: one other attendee and one far-end voice"
