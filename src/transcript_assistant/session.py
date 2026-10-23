"""One recording from start to a finished meta.json, shared by `record` and `watch`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .store import new_meeting_dir, write_json_atomic


@dataclass
class Session:
    folder: Path
    meta: dict
    recording: object  # recorder.Recording
    live: object = None  # live.LiveSpeakers while naming voices mid-call
    transcriber: object = None  # live_transcribe.LiveTranscriber while tracking questions mid-call


def start_session(root: Path, title: str, extra_meta: dict | None = None, recording_factory=None,
                  live_factory=None, transcriber_factory=None) -> Session:
    """`live_factory(folder)` returns a LiveSpeakers to run on the far-end audio, or None.
    `transcriber_factory(folder)` returns a LiveTranscriber to run on both channels, or None."""
    started = datetime.now(timezone.utc)
    folder = new_meeting_dir(root, title, started)
    meta = {
        "title": title,
        "started_at": started.isoformat(),
        "status": "recording",
        "tool_version": __version__,
        **(extra_meta or {}),
    }
    write_json_atomic(folder / "meta.json", meta)
    if recording_factory is None:
        from .recorder import Recording as recording_factory
    live = _optional(live_factory, folder, meta, "live speaker naming")
    transcriber = _optional(transcriber_factory, folder, meta, "live transcription")
    listeners = {}
    others = [f for f in (live and live.feed, transcriber and transcriber.feed_others) if f]
    if others:
        listeners["on_others_audio"] = others[0] if len(others) == 1 else _fan_out(others)
    if transcriber:
        listeners["on_mic_audio"] = transcriber.feed_mic
    rec = recording_factory(folder / "audio", **listeners)
    try:
        if live:
            live.start()
        if transcriber:
            transcriber.start()
        rec.start()
    except Exception as exc:
        for helper in (live, transcriber):
            if helper:
                helper.stop()
        meta.update(status="failed", warnings=[f"could not start capture: {type(exc).__name__}: {exc}"])
        write_json_atomic(folder / "meta.json", meta)
        raise
    return Session(folder, meta, rec, live, transcriber)


def _optional(factory, folder: Path, meta: dict, what: str):
    """Live helpers are conveniences; a failure is noted in the meeting's warnings, never fatal."""
    if factory is None:
        return None
    try:
        return factory(folder)
    except Exception as exc:
        meta.setdefault("calendar_warnings", []).append(f"{what} unavailable: {type(exc).__name__}: {exc}")
        return None


def _fan_out(listeners):
    def feed(block):
        for f in listeners:
            f(block)
    return feed


def finish_session(session: Session, stop_reason: str = "") -> dict:
    from .recorder import capture_warnings

    duration, stats = session.recording.stop()
    warnings = capture_warnings(duration, stats) + list(session.meta.get("calendar_warnings", []))
    # The live helpers are extras: if one fails here, the meeting must still be marked recorded
    # so it gets transcribed.
    if session.live is not None:
        try:
            session.live.stop()
            voices = session.live.tracker.voices.values()
            session.meta["live_voices"] = {"heard": len(voices), "named": sum(1 for v in voices if v.name)}
        except Exception as exc:
            warnings.append(f"live speaker naming failed at the end of the call: {type(exc).__name__}: {exc}")
        if session.live.errors:
            warnings.append(f"live speaker naming: {session.live.errors} window(s) could not be fingerprinted")
    if session.transcriber is not None:
        try:
            session.transcriber.stop()
            session.meta["live_questions"] = session.transcriber.write_summary()
        except Exception as exc:
            warnings.append(f"could not save live questions: {type(exc).__name__}: {exc}")
        if session.transcriber.errors:
            last = getattr(session.transcriber, "last_error", "")
            warnings.append(f"live transcription: {session.transcriber.errors} step(s) failed"
                            + (f" (last: {last})" if last else ""))
    session.meta.update(
        status="recorded",
        ended_at=datetime.now(timezone.utc).isoformat(),
        duration_s=round(duration, 1),
        stop_reason=stop_reason,
        channels=[asdict(s) for s in stats],
        capture_warnings=warnings,
        warnings=warnings,
    )
    write_json_atomic(session.folder / "meta.json", session.meta)
    return session.meta
