"""Local GPU transcription with faster-whisper."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .audio import WHISPER_RATE, load_channel
from .merge import Segment, merge, render_markdown
from .store import read_json, write_json_atomic

DEFAULT_MODEL = "large-v3-turbo"


def load_model(model_name: str = DEFAULT_MODEL, device: str = "cuda", compute_type: str = "float16"):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_channel(model, audio, channel: str, language: str | None = None) -> list[Segment]:
    if audio.size < WHISPER_RATE:  # under a second of audio
        return []
    segments, _info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,  # limits runaway repetition on long meetings
    )
    return [Segment(round(s.start, 2), round(s.end, 2), s.text.strip(), channel) for s in segments]


def transcribe_meeting(folder: Path, model, model_name: str = DEFAULT_MODEL,
                       language: str | None = None) -> dict:
    meta_path = folder / "meta.json"
    meta = read_json(meta_path)
    audio_dir = folder / "audio"

    per_channel = {}
    for channel in ("mic", "others"):
        audio = load_channel(audio_dir, channel)
        per_channel[channel] = transcribe_channel(model, audio, channel, language)

    segments, echo_dropped = merge(per_channel["mic"], per_channel["others"])
    warnings = list(meta.get("capture_warnings", []))  # recomputed, so re-runs never duplicate
    if not segments:
        warnings.append("transcription produced no text")

    write_json_atomic(folder / "transcript.json", {"segments": [s.to_dict() for s in segments]})
    md = render_markdown(meta.get("title", folder.name), meta.get("started_at", ""), segments, warnings)
    (folder / "transcript.md").write_text(md, encoding="utf-8")

    meta.update(
        status="transcribed",
        transcribed_at=datetime.now(timezone.utc).isoformat(),
        model=model_name,
        segment_counts={c: len(v) for c, v in per_channel.items()},
        echo_segments_dropped=echo_dropped,
        warnings=warnings,
    )
    write_json_atomic(meta_path, meta)
    return meta
