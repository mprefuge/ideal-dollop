"""Local, per-meeting folder layout and the index that lists them.

Layout under the root (default %USERPROFILE%\\MeetingTranscripts):

    index.json
    2026-10-15_0930_weekly-sync/
        meta.json
        transcript.md
        transcript.json
        audio/mic_0000.flac, audio/others_0000.flac, ...
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

INDEX_NAME = "index.json"


def default_root() -> Path:
    env = os.environ.get("TA_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / "MeetingTranscripts"


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "meeting"


def new_meeting_dir(root: Path, title: str, started: datetime) -> Path:
    """Create a fresh folder; never reuse an existing one."""
    base = f"{started.astimezone():%Y-%m-%d_%H%M}_{slugify(title)}"
    path = root / base
    n = 2
    while path.exists():
        path = root / f"{base}-{n}"
        n += 1
    (path / "audio").mkdir(parents=True)
    return path


def write_json_atomic(path: Path, data: object) -> None:
    """Write via temp file + rename so a crash never leaves half a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _replace(src: str, dst: Path, attempts: int = 40, wait_s: float = 0.05) -> None:
    """os.replace, retried while another process has the target open.

    On Windows a file that someone else is reading can't be replaced; os.replace
    raises PermissionError (WinError 5) until they close it. The question window
    reads questions.json and meta.json every second, and antivirus or the search
    indexer can hold a new file briefly, so a single attempt fails now and then.
    """
    import time

    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait_s)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rebuild_index(root: Path) -> list[dict]:
    """Rebuild index.json from every meeting's meta.json.

    The index is derived, never edited by hand, so it cannot drift from the
    folders it describes.
    """
    entries = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            meta = read_json(meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            entries.append({"folder": meta_path.parent.name, "error": f"unreadable meta.json: {exc}"})
            continue
        entries.append(
            {
                "folder": meta_path.parent.name,
                "title": meta.get("title"),
                "started_at": meta.get("started_at"),
                "duration_s": meta.get("duration_s"),
                "status": meta.get("status"),
                "warnings": meta.get("warnings", []),
            }
        )
    entries.sort(key=lambda e: e.get("started_at") or "", reverse=True)
    write_json_atomic(root / INDEX_NAME, {"meetings": entries})
    return entries


def audio_purge_candidates(root: Path, older_than_days: int, now: datetime | None = None) -> list[Path]:
    """Audio folders of transcribed meetings older than the cutoff.

    Meetings that were never transcribed are never candidates: their audio is
    the only copy of the meeting.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=older_than_days)
    out = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            meta = read_json(meta_path)
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") != "transcribed" or meta.get("keep_audio"):
            continue
        started = meta.get("started_at")
        if not started or datetime.fromisoformat(started) > cutoff:
            continue
        audio = meta_path.parent / "audio"
        if audio.is_dir() and any(audio.iterdir()):
            out.append(audio)
    return out


def purge_audio(dirs: list[Path]) -> None:
    for d in dirs:
        shutil.rmtree(d)
        meta_path = d.parent / "meta.json"
        meta = read_json(meta_path)
        meta["audio_purged_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(meta_path, meta)
