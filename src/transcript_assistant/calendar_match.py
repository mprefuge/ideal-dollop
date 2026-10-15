"""Read-only calendar lookup from published ICS links, to name and annotate recordings."""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

EARLY_JOIN = timedelta(minutes=10)
PLATFORM_HINTS = {
    "Zoom": ("zoom.us",),
    "Teams": ("teams.microsoft.com", "teams.live.com"),
    "Webex": ("webex.com",),
    "Browser": ("meet.google.com", "teams.microsoft.com", "zoom.us"),
    "Slack": ("slack.com",),
}


@dataclass
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
    uid: str = ""
    organizer: str = ""
    attendees: list[str] = field(default_factory=list)
    location: str = ""
    description: str = ""

    def to_meta(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        d.pop("description")  # can be long and hold dial-in secrets; title and attendees are enough
        return d

    def text(self) -> str:
        return f"{self.title}\n{self.location}\n{self.description}".lower()


def _addr(value) -> str:
    name = value.params.get("CN") if hasattr(value, "params") else None
    email = str(value).removeprefix("mailto:").removeprefix("MAILTO:")
    return f"{name} <{email}>" if name and name != email else email


def _aware(dt) -> datetime | None:
    if not isinstance(dt, datetime):  # all-day events are dates; never meetings we record
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_events(ics_text: str | bytes, window_start: datetime, window_end: datetime) -> list[CalendarEvent]:
    """Expand recurring events and return the timed events overlapping the window."""
    import icalendar
    import recurring_ical_events

    cal = icalendar.Calendar.from_ical(ics_text)
    out = []
    for comp in recurring_ical_events.of(cal).between(window_start, window_end):
        if str(comp.get("STATUS", "")).upper() == "CANCELLED":
            continue
        start = _aware(comp.decoded("DTSTART"))
        end = _aware(comp.decoded("DTEND")) if comp.get("DTEND") else None
        if start is None:
            continue
        if end is None:
            dur = comp.decoded("DURATION") if comp.get("DURATION") else timedelta(hours=1)
            end = start + dur
        attendees = comp.get("ATTENDEE", [])
        if not isinstance(attendees, list):
            attendees = [attendees]
        out.append(CalendarEvent(
            title=str(comp.get("SUMMARY", "")).strip() or "Meeting",
            start=start, end=end,
            uid=str(comp.get("UID", "")),
            organizer=_addr(comp["ORGANIZER"]) if comp.get("ORGANIZER") else "",
            attendees=[_addr(a) for a in attendees],
            location=str(comp.get("LOCATION", "")),
            description=str(comp.get("DESCRIPTION", "")),
        ))
    return out


def best_match(events: list[CalendarEvent], now: datetime, platforms: list[str]) -> CalendarEvent | None:
    """The event this recording most likely belongs to, or None.

    Candidates are events in progress (or starting within ten minutes). Among
    several, prefer one whose link or location names the platform in use, then
    the one that started most recently.
    """
    live = [e for e in events if e.start - EARLY_JOIN <= now < e.end]
    if not live:
        return None
    hints = tuple(h for p in platforms for h in PLATFORM_HINTS.get(p, ()))

    def score(e: CalendarEvent):
        names_platform = bool(hints) and any(h in e.text() for h in hints)
        return (names_platform, e.start)

    return max(live, key=score)


def should_skip(event: CalendarEvent | None, skip_keywords: list[str]) -> bool:
    if event is None:
        return False
    text = event.text()
    return any(k.lower() in text for k in skip_keywords)


class CalendarSource:
    """Fetches each ICS link at most every `refresh_minutes`, falling back to the last good copy."""

    def __init__(self, urls: list[str], cache_dir: Path, refresh_minutes: int = 15, fetch=None):
        self.urls = urls
        self.cache_dir = cache_dir
        self.refresh_s = refresh_minutes * 60
        self._fetch = fetch or self._http_get
        self._last_fetch: dict[str, float] = {}
        self.errors: list[str] = []

    @staticmethod
    def _http_get(url: str) -> bytes:  # pragma: no cover - network
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read()

    def _cache_path(self, url: str) -> Path:
        # Hash the URL: secret ICS links must not appear in file names or logs.
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.ics"

    def _text(self, url: str) -> bytes | None:
        path = self._cache_path(url)
        stale = time.monotonic() - self._last_fetch.get(url, -1e18) >= self.refresh_s
        if stale:
            try:
                data = self._fetch(url)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                self._last_fetch[url] = time.monotonic()
            except Exception as exc:
                self.errors.append(f"calendar {path.stem}: fetch failed ({type(exc).__name__}); using cached copy")
                log.warning(self.errors[-1])
        return path.read_bytes() if path.exists() else None

    def events_now(self, now: datetime) -> list[CalendarEvent]:
        self.errors = []
        events: list[CalendarEvent] = []
        for url in self.urls:
            data = self._text(url)
            if data is None:
                continue
            try:
                events += parse_events(data, now - timedelta(hours=12), now + timedelta(hours=1))
            except Exception as exc:
                self.errors.append(f"calendar {self._cache_path(url).stem}: unreadable ICS ({exc})")
                log.warning(self.errors[-1])
        return events
