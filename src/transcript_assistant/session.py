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


def start_session(root: Path, title: str, extra_meta: dict | None = None, recording_factory=None) -> Session:
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
    rec = recording_factory(folder / "audio")
    try:
        rec.start()
    except Exception as exc:
        meta.update(status="failed", warnings=[f"could not start capture: {type(exc).__name__}: {exc}"])
        write_json_atomic(folder / "meta.json", meta)
        raise
    return Session(folder, meta, rec)


def finish_session(session: Session, stop_reason: str = "") -> dict:
    from .recorder import capture_warnings

    duration, stats = session.recording.stop()
    warnings = capture_warnings(duration, stats) + list(session.meta.get("calendar_warnings", []))
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
