"""Local GPU transcription with faster-whisper."""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

from .audio import WHISPER_RATE, load_channel
from .merge import Segment, merge, render_markdown
from .store import read_json, write_json_atomic

DEFAULT_MODEL = "large-v3-turbo"


def load_model(model_name: str = DEFAULT_MODEL, device: str = "cuda", compute_type: str = "float16"):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_channel(model, audio, channel: str, language: str | None = None,
                       word_timestamps: bool = False) -> list[Segment]:
    if audio.size < WHISPER_RATE:  # under a second of audio
        return []
    segments, _info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,  # limits runaway repetition on long meetings
        word_timestamps=word_timestamps,
    )
    out = []
    for s in segments:
        words = [(w.start, w.end, w.word) for w in (getattr(s, "words", None) or [])]
        out.append(Segment(round(s.start, 2), round(s.end, 2), s.text.strip(), channel, words=words))
    return out


def write_transcript(folder: Path, meta: dict, segments: list[Segment], warnings: list[str]) -> None:
    """Write transcript.json and transcript.md, applying meta['speaker_names']."""
    names = meta.get("speaker_names", {})
    write_json_atomic(folder / "transcript.json", {"segments": [s.to_dict(names) for s in segments]})
    md = render_markdown(meta.get("title", folder.name), meta.get("started_at", ""), segments, warnings, names)
    (folder / "transcript.md").write_text(md, encoding="utf-8")


def transcribe_meeting(folder: Path, model, model_name: str = DEFAULT_MODEL,
                       language: str | None = None, diarizer: Callable | None = None,
                       my_name: str = "Me", my_emails: list[str] | None = None) -> dict:
    """Transcribe both channels, optionally split the far end by speaker, and write outputs.

    `diarizer` takes 16 kHz far-end audio and returns diarize.Turn objects.
    """
    from .diarize import assign_speakers, calendar_names

    meta_path = folder / "meta.json"
    meta = read_json(meta_path)
    audio_dir = folder / "audio"
    warnings = list(meta.get("capture_warnings", []))  # recomputed, so re-runs never duplicate

    per_channel = {}
    for channel in ("mic", "others"):
        audio = load_channel(audio_dir, channel)
        diarize_this = diarizer is not None and channel == "others"
        segs = transcribe_channel(model, audio, channel, language, word_timestamps=diarize_this)
        if diarize_this and segs:
            try:
                segs = assign_speakers(segs, diarizer(audio))
                meta["diarized"] = True
            except Exception as exc:
                warnings.append(f"speaker separation failed, far end left as one speaker: "
                                f"{type(exc).__name__}: {exc}")
                meta["diarized"] = False
        per_channel[channel] = segs

    segments, echo_dropped = merge(per_channel["mic"], per_channel["others"])
    if not segments:
        warnings.append("transcription produced no text")

    # Names: automatic ones first, then anything the user set with `rename`, which always wins.
    auto, source = calendar_names(segments, meta.get("calendar_event"), my_emails or [])
    if my_name and my_name != "Me":
        auto["Me"] = my_name
    user_set = meta.get("speaker_names_user", {})
    meta["speaker_names"] = {**auto, **user_set}
    meta["speaker_names_source"] = source or None

    write_transcript(folder, meta, segments, warnings)

    meta.update(
        status="transcribed",
        transcribed_at=datetime.now(timezone.utc).isoformat(),
        model=model_name,
        segment_counts={c: len(v) for c, v in per_channel.items()},
        speakers=sorted({s.speaker for s in segments}),
        echo_segments_dropped=echo_dropped,
        warnings=warnings,
    )
    write_json_atomic(meta_path, meta)
    return meta
