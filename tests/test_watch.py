from datetime import datetime, timedelta, timezone

import pytest

from transcript_assistant.calendar_match import CalendarEvent, CalendarSource, best_match, parse_events, should_skip
from transcript_assistant.config import Config, load_config, write_default_config
from transcript_assistant.detector import MicUse, active_meeting_apps, classify
from transcript_assistant.recorder import ChannelStats
from transcript_assistant.store import read_json, write_json_atomic
from transcript_assistant.watcher import (IDLE, RECORDING, SUPPRESSED, Watcher, pending_meetings,
                                          recover_interrupted)

# --- detector --------------------------------------------------------------

ZOOM = r"C:#Users#micah#AppData#Roaming#Zoom#bin#Zoom.exe"
CHROME = r"C:#Program Files#Google#Chrome#Application#chrome.exe"
PYTHON = r"C:#Users#micah#ta#.venv#Scripts#pythonw.exe"
TEAMS = "MSTeams_8wekyb3d8bbwe"


def test_classify_known_apps_and_ignores_self():
    assert classify(ZOOM) == "Zoom"
    assert classify(TEAMS) == "Teams"
    assert classify(CHROME) == "Browser"
    assert classify(CHROME, include_browsers=False) is None
    assert classify(PYTHON) is None  # our own recorder must never keep a meeting "alive"
    assert classify(r"C:#Program Files#Audacity#audacity.exe") is None
    assert classify(r"C:#x#bluejeans.exe", extra_apps={"bluejeans": "BlueJeans"}) == "BlueJeans"


def test_only_apps_holding_the_mic_now_count():
    uses = [MicUse(ZOOM, True), MicUse(TEAMS, False), MicUse(PYTHON, True), MicUse(CHROME, True)]
    assert active_meeting_apps(uses) == ["Browser", "Zoom"]


# --- calendar --------------------------------------------------------------

ICS = b"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:test
BEGIN:VEVENT
UID:weekly-1
SUMMARY:Partner Integrations sync
DTSTART:20261001T140000Z
DTEND:20261001T143000Z
RRULE:FREQ=WEEKLY;BYDAY=TH
LOCATION:https://teams.microsoft.com/l/meetup-join/abc
ORGANIZER;CN=Dana Lee:mailto:dana@example.edu
ATTENDEE;CN=Micah:mailto:micah@example.edu
ATTENDEE:mailto:sam@example.edu
END:VEVENT
BEGIN:VEVENT
UID:overlap-1
SUMMARY:Donor call
DTSTART:20261015T140000Z
DTEND:20261015T150000Z
LOCATION:https://zoom.us/j/123
END:VEVENT
BEGIN:VEVENT
UID:allday
SUMMARY:Conference
DTSTART;VALUE=DATE:20261015
DTEND;VALUE=DATE:20261016
END:VEVENT
BEGIN:VEVENT
UID:cancelled
SUMMARY:Cancelled thing
DTSTART:20261015T140000Z
DTEND:20261015T150000Z
STATUS:CANCELLED
END:VEVENT
BEGIN:VEVENT
UID:private
SUMMARY:Pastoral care [no record]
DTSTART:20261015T170000Z
DTEND:20261015T180000Z
END:VEVENT
END:VCALENDAR
"""

NOW = datetime(2026, 10, 15, 14, 5, tzinfo=timezone.utc)  # a Thursday


def events_at(now):
    return parse_events(ICS, now - timedelta(hours=12), now + timedelta(hours=1))


def test_recurring_event_expands_and_all_day_and_cancelled_are_ignored():
    titles = sorted(e.title for e in events_at(NOW))
    assert titles == ["Donor call", "Partner Integrations sync"]
    sync = next(e for e in events_at(NOW) if e.uid == "weekly-1")
    assert sync.start == datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
    assert sync.organizer == "Dana Lee <dana@example.edu>"
    assert sync.attendees == ["Micah <micah@example.edu>", "sam@example.edu"]


def test_best_match_prefers_the_platform_in_use():
    evs = events_at(NOW)
    assert best_match(evs, NOW, ["Zoom"]).title == "Donor call"
    assert best_match(evs, NOW, ["Teams"]).title == "Partner Integrations sync"


def test_best_match_allows_joining_early_but_not_after_the_end():
    evs = events_at(NOW)
    assert best_match(evs, datetime(2026, 10, 15, 13, 52, tzinfo=timezone.utc), ["Teams"]) is not None
    assert best_match(evs, datetime(2026, 10, 15, 15, 1, tzinfo=timezone.utc), ["Zoom"]) is None


def test_skip_keyword():
    t = datetime(2026, 10, 15, 17, 10, tzinfo=timezone.utc)
    ev = best_match(events_at(t), t, ["Teams"])
    assert ev.title.startswith("Pastoral care")
    assert should_skip(ev, ["[no record]"])
    assert not should_skip(None, ["[no record]"])


def test_event_meta_omits_description():
    ev = CalendarEvent("t", NOW, NOW, description="dial-in PIN 1234")
    assert "description" not in ev.to_meta()


def test_calendar_source_falls_back_to_cache_and_hides_url(tmp_path):
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("offline")
        return ICS

    secret = "https://outlook.office365.com/owa/calendar/SECRET/reachcalendar.ics"
    src = CalendarSource([secret], tmp_path, refresh_minutes=0, fetch=fetch)
    assert len(src.events_now(NOW)) == 2
    assert len(src.events_now(NOW)) == 2  # second fetch fails, cache used
    assert src.errors and "SECRET" not in src.errors[0]
    assert all("SECRET" not in p.name for p in tmp_path.iterdir())


# --- config ----------------------------------------------------------------

def test_config_defaults_roundtrip_and_rejects_typos(tmp_path):
    path, created = write_default_config(tmp_path)
    assert created and load_config(tmp_path) == Config()
    assert write_default_config(tmp_path) == (path, False)
    write_json_atomic(path, {"ics_url": "typo"})
    with pytest.raises(ValueError, match="ics_url"):
        load_config(tmp_path)


# --- watcher state machine -------------------------------------------------

class FakeRecording:
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.started = self.stopped = False
        self.fail = False
        self.fail_all = True

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return 600.0, [ChannelStats("others", frames=600 * 48000, peak=0.5),
                       ChannelStats("mic", frames=600 * 48000, peak=0.5)]

    def any_failed(self):
        return self.fail

    def all_failed(self):
        return self.fail and self.fail_all

    def failure_detail(self):
        return "mic: OSError: device gone" if self.fail else ""


class Harness:
    def __init__(self, tmp_path, calendar=None, cfg=None):
        self.t = 0.0
        self.uses: list[MicUse] = []
        self.recorded = []
        self.notes = []
        self.recordings = []

        def factory(audio_dir):
            r = FakeRecording(audio_dir)
            self.recordings.append(r)
            return r

        self.w = Watcher(tmp_path, cfg or Config(), lambda: self.uses, calendar,
                         on_recorded=self.recorded.append, notify=self.notes.append,
                         clock=lambda: self.t, wall_clock=lambda: NOW, recording_factory=factory)

    def advance(self, seconds, step=1.0):
        end = self.t + seconds
        while self.t < end:
            self.t += step
            self.w.tick()


def test_starts_after_debounce_and_stops_after_grace(tmp_path):
    h = Harness(tmp_path)
    h.uses = [MicUse(ZOOM, True)]
    h.advance(3)
    assert h.w.state == IDLE  # a brief mic test doesn't start a recording
    h.advance(3)
    assert h.w.state == RECORDING
    folder = h.w.session.folder
    assert folder.name.endswith("zoom-meeting")
    h.uses = []
    h.advance(30)
    assert h.w.state == RECORDING  # brief drop (rejoin, device switch) keeps recording
    h.uses = [MicUse(ZOOM, True)]
    h.advance(2)
    h.uses = []
    h.advance(61)
    assert h.w.state == IDLE
    assert h.recorded == [folder]
    meta = read_json(folder / "meta.json")
    assert meta["status"] == "recorded" and meta["platforms"] == ["Zoom"] and meta["trigger"] == "auto"
    assert meta["stop_reason"] == "meeting app released the microphone"
    assert len(h.recordings) == 1


def test_calendar_title_and_attendees_land_in_meta(tmp_path):
    src = CalendarSource(["u"], tmp_path / ".cache", fetch=lambda url: ICS)
    h = Harness(tmp_path, calendar=src)
    h.uses = [MicUse(TEAMS, True)]
    h.advance(6)
    meta = h.w.session.meta
    assert meta["title"] == "Partner Integrations sync"
    assert meta["calendar_event"]["attendees"] == ["Micah <micah@example.edu>", "sam@example.edu"]


def test_manual_stop_does_not_restart_same_meeting(tmp_path):
    h = Harness(tmp_path)
    h.uses = [MicUse(ZOOM, True)]
    h.advance(6)
    h.w.stop_now()
    assert h.w.state == SUPPRESSED
    h.advance(120)
    assert h.w.state == SUPPRESSED and len(h.recordings) == 1
    h.uses = []
    h.advance(2)
    assert h.w.state == IDLE
    h.uses = [MicUse(ZOOM, True)]
    h.advance(6)
    assert h.w.state == RECORDING and len(h.recordings) == 2  # next meeting records


def test_skip_keyword_meeting_is_not_recorded(tmp_path):
    src = CalendarSource(["u"], tmp_path / ".cache", fetch=lambda url: ICS)
    h = Harness(tmp_path, calendar=src)
    h.w.wall_clock = lambda: datetime(2026, 10, 15, 17, 10, tzinfo=timezone.utc)
    h.uses = [MicUse(TEAMS, True)]
    h.advance(10)
    assert h.w.state == SUPPRESSED and h.recordings == []
    assert not list(tmp_path.glob("*/meta.json"))


def test_pause_blocks_new_recordings_until_it_expires(tmp_path):
    h = Harness(tmp_path)
    h.w.pause(300)
    h.uses = [MicUse(ZOOM, True)]
    h.advance(60)
    assert h.w.state == IDLE and h.w.status == "Paused"
    h.advance(250)
    assert h.w.state == RECORDING


def test_max_duration_stops_and_suppresses(tmp_path):
    h = Harness(tmp_path, cfg=Config(max_minutes=1))
    h.uses = [MicUse(ZOOM, True)]
    h.advance(70)
    assert h.w.state == SUPPRESSED and len(h.recorded) == 1


def test_channel_failure_stops_recording(tmp_path):
    h = Harness(tmp_path)
    h.uses = [MicUse(ZOOM, True)]
    h.advance(6)
    h.recordings[0].fail = True
    h.advance(1)
    assert h.w.state == SUPPRESSED
    assert read_json(h.recorded[0] / "meta.json")["stop_reason"] == "capture channel failed (mic: OSError: device gone)"


def test_one_failed_channel_keeps_recording_the_other(tmp_path):
    h = Harness(tmp_path)
    h.uses = [MicUse(ZOOM, True)]
    h.advance(6)
    h.recordings[0].fail, h.recordings[0].fail_all = True, False
    h.advance(5)
    assert h.w.state == RECORDING
    assert sum("still recording the other side" in n for n in h.notes) == 1  # warned once, not every tick


def test_mic_read_error_is_survivable(tmp_path):
    h = Harness(tmp_path)

    def boom():
        raise OSError("registry unavailable")

    h.w.read_mic = boom
    h.advance(3)
    assert h.w.state == IDLE


def test_shutdown_finalizes_an_active_recording(tmp_path):
    h = Harness(tmp_path)
    h.uses = [MicUse(ZOOM, True)]
    h.advance(6)
    h.w.shutdown()
    assert read_json(h.recorded[0] / "meta.json")["stop_reason"] == "watcher exited"


def test_interrupted_recordings_are_recovered_for_transcription(tmp_path):
    folder = tmp_path / "2026-10-15_1400_crashed"
    folder.mkdir()
    write_json_atomic(folder / "meta.json", {"title": "x", "status": "recording"})
    assert recover_interrupted(tmp_path) == [folder]
    meta = read_json(folder / "meta.json")
    assert meta["status"] == "recorded" and "interrupted" in meta["warnings"][0]
    assert pending_meetings(tmp_path) == [folder]
