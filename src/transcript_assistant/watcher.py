"""Background loop: start recording when a meeting app opens the mic, stop when it lets go."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .calendar_match import CalendarSource, best_match, should_skip
from .config import Config
from .detector import MicUse, active_meeting_apps
from .session import Session, finish_session, start_session
from .store import read_json, rebuild_index, write_json_atomic

log = logging.getLogger(__name__)

IDLE, RECORDING, SUPPRESSED = "idle", "recording", "suppressed"


class Watcher:
    """A pure state machine around injected I/O, so it can be tested without devices.

    SUPPRESSED means "a meeting is still holding the mic but we chose not to
    record it" (stopped from the tray, skipped by keyword, hit the time limit,
    or capture failed). It returns to IDLE once the mic is released, so the
    same meeting is never restarted behind the user's back.
    """

    def __init__(self, root: Path, cfg: Config, read_mic: Callable[[], list[MicUse]],
                 calendar: CalendarSource | None = None,
                 on_recorded: Callable[[Path], None] = lambda folder: None,
                 notify: Callable[[str], None] = lambda msg: None,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 recording_factory=None, live_factory=None):
        self.root = root
        self.live_factory = live_factory
        self.cfg = cfg
        self.read_mic = read_mic
        self.calendar = calendar
        self.on_recorded = on_recorded
        self.notify = notify
        self.clock = clock
        self.wall_clock = wall_clock
        self.recording_factory = recording_factory
        self.state = IDLE
        self.session: Session | None = None
        self.paused_until = 0.0
        self._held_since: float | None = None
        self._last_seen = 0.0
        self._started = 0.0
        self._lock = threading.RLock()

    # --- loop ------------------------------------------------------------

    def tick(self) -> None:
        with self._lock:
            try:
                apps = active_meeting_apps(self.read_mic(), self.cfg.include_browsers, self.cfg.extra_apps)
            except Exception:
                log.exception("could not read microphone use")
                return
            t = self.clock()
            if self.state == RECORDING:
                if apps:
                    self._last_seen = t
                if self.session and self.session.recording.any_failed():
                    detail = self.session.recording.failure_detail()
                    self.notify(f"Recording stopped: an audio channel failed ({detail})")
                    self._stop(f"capture channel failed ({detail})", next_state=SUPPRESSED)
                elif not apps and t - self._last_seen >= self.cfg.stop_after_seconds:
                    self._stop("meeting app released the microphone", next_state=IDLE)
                elif self.cfg.max_minutes and t - self._started >= self.cfg.max_minutes * 60:
                    self._stop(f"reached the {self.cfg.max_minutes}-minute limit", next_state=SUPPRESSED)
            elif self.state == SUPPRESSED:
                if not apps:
                    self.state = IDLE
                    self._held_since = None
            else:
                if not apps:
                    self._held_since = None
                    return
                if t < self.paused_until:
                    return
                if self._held_since is None:
                    self._held_since = t
                if t - self._held_since >= self.cfg.start_after_seconds:
                    self._begin(apps, t)

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self.tick()
            stop_event.wait(self.cfg.poll_seconds)
        self.shutdown()

    # --- actions ---------------------------------------------------------

    def _begin(self, apps: list[str], t: float) -> None:
        event, cal_warnings = None, []
        if self.calendar is not None:
            try:
                events = self.calendar.events_now(self.wall_clock())
                cal_warnings = list(self.calendar.errors)
                event = best_match(events, self.wall_clock(), apps)
            except Exception as exc:
                cal_warnings = [f"calendar lookup failed: {type(exc).__name__}: {exc}"]
                log.exception("calendar lookup failed")
        if should_skip(event, self.cfg.skip_keywords):
            log.info("Not recording %r: calendar event has a skip keyword", event.title)
            self.notify(f"Not recording \"{event.title}\" (marked no-record)")
            self.state = SUPPRESSED
            return

        title = event.title if event else f"{apps[0]} meeting"
        extra = {
            "platforms": apps,
            "trigger": "auto",
            "consent_confirmed": "standing acknowledgement given when watch started",
            "calendar_event": event.to_meta() if event else None,
            "calendar_warnings": cal_warnings,
        }
        try:
            self.session = start_session(self.root, title, extra, self.recording_factory, self.live_factory)
        except Exception as exc:
            log.exception("could not start recording")
            self.notify(f"Could not start recording: {exc}")
            self.state = SUPPRESSED
            return
        self.state = RECORDING
        self._started = self._last_seen = t
        log.info("Recording %s (%s) to %s", title, ", ".join(apps), self.session.folder)
        self.notify(f"Recording: {title}")

    def _stop(self, reason: str, next_state: str) -> None:
        session, self.session = self.session, None
        self.state = next_state
        self._held_since = None
        if session is None:
            return
        try:
            meta = finish_session(session, reason)
        except Exception:
            log.exception("failed to finalize %s", session.folder)
            return
        log.info("Stopped %s: %s", session.folder.name, reason)
        self.notify(f"Stopped: {meta['title']}" + (f" ({len(meta['warnings'])} warning(s))" if meta["warnings"] else ""))
        rebuild_index(self.root)
        self.on_recorded(session.folder)

    def stop_now(self) -> None:
        """From the tray: stop and don't restart until this meeting releases the mic."""
        with self._lock:
            if self.state == RECORDING:
                self._stop("stopped by user", next_state=SUPPRESSED)

    def pause(self, seconds: float) -> None:
        with self._lock:
            self.paused_until = self.clock() + seconds
            if self.state == RECORDING:
                self._stop("paused by user", next_state=SUPPRESSED)

    def resume(self) -> None:
        with self._lock:
            self.paused_until = 0.0

    def shutdown(self) -> None:
        with self._lock:
            if self.state == RECORDING:
                self._stop("watcher exited", next_state=IDLE)

    @property
    def status(self) -> str:
        with self._lock:
            if self.state == RECORDING and self.session:
                return f"Recording: {self.session.meta['title']}"
            if self.clock() < self.paused_until:
                return "Paused"
            if self.state == SUPPRESSED:
                return "Not recording this meeting"
            return "Watching for meetings"


def recover_interrupted(root: Path) -> list[Path]:
    """Meetings left in 'recording' by a crash or power loss become 'recorded' so they get transcribed."""
    out = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        if meta.get("status") != "recording":
            continue
        warn = "recording was interrupted (crash or shutdown); the end of the meeting may be missing"
        meta["status"] = "recorded"
        meta["capture_warnings"] = list(meta.get("capture_warnings", [])) + [warn]
        meta["warnings"] = list(meta["capture_warnings"])
        write_json_atomic(meta_path, meta)
        out.append(meta_path.parent)
    return out


def pending_meetings(root: Path) -> list[Path]:
    out = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            if read_json(meta_path).get("status") == "recorded":
                out.append(meta_path.parent)
        except Exception:
            continue
    return out


class TranscriptionWorker(threading.Thread):
    """Transcribes finished recordings one at a time; loads the model on first use."""

    def __init__(self, root: Path, load_model: Callable[[], object], transcribe: Callable[[Path, object], dict],
                 notify: Callable[[str], None] = lambda msg: None,
                 after: Callable[[Path], object] | None = None):
        super().__init__(name="transcriber", daemon=True)
        self.after = after
        self.root = root
        self.load_model = load_model
        self.transcribe = transcribe
        self.notify = notify
        self.queue: queue.Queue[Path | None] = queue.Queue()
        self._model = None

    def submit(self, folder: Path) -> None:
        self.queue.put(folder)

    def run(self) -> None:
        while True:
            folder = self.queue.get()
            if folder is None:
                return
            try:
                if self._model is None:
                    self._model = self.load_model()
                meta = self.transcribe(folder, self._model)
                self.notify(f"Transcript ready: {meta.get('title', folder.name)}")
                if self.after is not None:
                    try:
                        self.after(folder)
                    except Exception:
                        log.exception("post-transcription step failed for %s", folder)
            except Exception as exc:
                # Audio is untouched and status stays 'recorded', so `transcribe --pending` retries it.
                log.exception("transcription failed for %s", folder)
                self.notify(f"Transcription failed for {folder.name}: {exc}")
            finally:
                rebuild_index(self.root)
