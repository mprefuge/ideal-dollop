"""User settings stored as config.json in the transcripts root."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .store import write_json_atomic

CONFIG_NAME = "config.json"


@dataclass
class Config:
    # Secret ICS links (Outlook: Settings > Calendar > Shared calendars > Publish;
    # Google: calendar settings > "Secret address in iCal format"). Read-only.
    ics_urls: list[str] = field(default_factory=list)
    calendar_refresh_minutes: int = 15
    # A meeting whose calendar title or description contains one of these is not recorded.
    skip_keywords: list[str] = field(default_factory=lambda: ["[no record]", "#norecord"])
    include_browsers: bool = True
    # Extra app-id substrings to treat as meeting apps, e.g. {"bluejeans": "BlueJeans"}.
    extra_apps: dict[str, str] = field(default_factory=dict)
    start_after_seconds: int = 5     # mic must be held this long before recording starts
    stop_after_seconds: int = 60     # mic must be released this long before recording stops
    max_minutes: int = 240
    poll_seconds: float = 2.0
    language: str | None = None
    # How you appear in transcripts, and your addresses so calendar matching can tell you apart.
    my_name: str = "Me"
    my_emails: list[str] = field(default_factory=list)
    # Separate individual speakers on the far end (needs `pip install -e .[diarize]` and HF_TOKEN).
    diarize: bool = False
    # After a diarized meeting is transcribed, pop up a window to name its new speakers.
    prompt_for_names: bool = True
    # With diarize on: pop up a naming window during the call when a new far-end voice is heard.
    live_naming: bool = True
    # Part of a device name (see `devices`) to record instead of the Windows defaults. With
    # Bluetooth earbuds, calls often use the "Hands-Free" device while the default is "Stereo".
    speaker_device: str = ""
    mic_device: str = ""
    # How alike two voice fingerprints must be (cosine, 0-1) to count as the same person.
    voice_match_threshold: float = 0.55


def load_config(root: Path) -> Config:
    path = root / CONFIG_NAME
    if not path.exists():
        return Config()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {', '.join(unknown)}")
    return Config(**raw)


def write_default_config(root: Path) -> tuple[Path, bool]:
    """Write defaults only if no config exists. Returns (path, created)."""
    path = root / CONFIG_NAME
    if path.exists():
        return path, False
    write_json_atomic(path, asdict(Config()))
    return path, True
