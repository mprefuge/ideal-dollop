import json
import time

import numpy as np
import pytest

from transcript_assistant.audio import CAPTURE_RATE, WHISPER_RATE, ChunkWriter
from transcript_assistant.diarize import Turn
from transcript_assistant.live import (LiveSpeakers, VoiceTracker, absorb_late_answers, live_names,
                                       match_final_speakers, write_answer)
from transcript_assistant.recorder import ChannelStats
from transcript_assistant.session import finish_session, start_session
from transcript_assistant.store import read_json, write_json_atomic
from transcript_assistant.transcribe import transcribe_meeting

from test_speakers import WordModel


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


A, B = unit(1, 0, 0), unit(0, 1, 0)
SILENT = np.zeros(10, dtype=np.float32)


# --- tracker logic ----------------------------------------------------------

def test_new_voice_needs_three_consistent_windows():
    t = VoiceTracker()
    assert t.observe(0, A, SILENT) == (None, None)
    assert t.observe(3, A, SILENT) == (None, None)
    vid, voice = t.observe(6, A, SILENT)
    assert vid == "Voice 1" and voice.count == 3
    assert t.observe(9, A, SILENT) == ("Voice 1", None)  # known voice: no new prompt


def test_one_off_odd_windows_never_become_a_voice():
    t = VoiceTracker()
    for i, emb in enumerate([A, B, unit(0, 0, 1), unit(1, 1, 0), unit(0, 1, 1)]):
        assert t.observe(i * 3, emb, SILENT)[1] is None
    assert t.voices == {}


def test_pending_windows_expire():
    t = VoiceTracker()
    t.observe(0, A, SILENT)
    t.observe(3, A, SILENT)
    assert t.observe(60, A, SILENT) == (None, None)  # earlier two are too old to count


def test_naming_a_voice_like_another_merges_them():
    t = VoiceTracker()
    for i in range(3):
        t.observe(i, A, SILENT)
    for i in range(3):
        t.observe(10 + i, B, SILENT)
    t.set_name("Voice 1", "Dana")
    t.set_name("Voice 2", "Dana")
    assert t.voices["Voice 2"].name == "Dana"
    assert list(t.named_centroids()) == ["Dana"]
    restored = VoiceTracker.from_json(json.loads(json.dumps(t.to_json())))
    assert restored.voices["Voice 1"].name == "Dana"


# --- live session -----------------------------------------------------------

def tone(freq, seconds, rate=CAPTURE_RATE, amp=0.3):
    n = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * n)).astype(np.float32)


def freq_embed(audio16k):
    """Fake fingerprint: which of three pitch bands dominates."""
    spec = np.abs(np.fft.rfft(audio16k))
    peak = np.argmax(spec) * WHISPER_RATE / audio16k.size
    return {0: A, 1: B}.get(int(peak > 500), A) if peak < 1500 else unit(0, 0, 1)


def run_live(tmp_path, blocks, names=None):
    opened = []
    live = LiveSpeakers(tmp_path, freq_embed, on_new_voice=lambda f, vid: opened.append(vid))
    live.start()
    for b in blocks:
        for i in range(0, b.size, CAPTURE_RATE // 2):  # 0.5 s blocks, as the recorder sends
            live.feed(b[i:i + CAPTURE_RATE // 2])
    live.stop()
    return live, opened


def test_live_session_announces_each_new_voice_once_and_saves_a_clip(tmp_path):
    blocks = [tone(300, 12), np.zeros(6 * CAPTURE_RATE, dtype=np.float32), tone(800, 12), tone(300, 6)]
    live, opened = run_live(tmp_path, blocks)
    assert opened == ["Voice 1", "Voice 2"]
    assert (tmp_path / "live" / "voice_1.wav").exists() and (tmp_path / "live" / "voice_2.wav").exists()
    state = read_json(tmp_path / "live" / "voices.json")
    assert state["voices"]["Voice 2"]["first_seen_s"] == pytest.approx(18.0)  # silence still counts as time


def test_embedding_errors_are_counted_not_fatal(tmp_path):
    def broken(_a):
        raise RuntimeError("no GPU")

    live = LiveSpeakers(tmp_path, broken)
    live.start()
    live.feed(tone(300, 7))
    live.stop()
    assert live.errors == 2


def test_answers_written_by_the_window_are_applied(tmp_path):
    live, _ = run_live(tmp_path, [tone(300, 12)])
    write_answer(tmp_path, "Voice 1", "Dana Lee")
    absorb_late_answers(tmp_path)  # answered after the call ended
    assert read_json(tmp_path / "live" / "voices.json")["voices"]["Voice 1"]["name"] == "Dana Lee"
    assert list(live_names(tmp_path)) == ["Dana Lee"]
    assert not list((tmp_path / "live" / "answers").iterdir())


# --- carrying names into the transcript ---------------------------------------

def test_match_final_speakers_by_fingerprint():
    audio = {"Speaker 1": np.ones(WHISPER_RATE * 2), "Speaker 2": np.ones(WHISPER_RATE * 2) * 2,
             "Speaker 3": np.ones(100)}
    embed = lambda a: A if a[0] == 1 else B  # noqa: E731
    named = {"Dana": A, "Sam": unit(0.1, 1, 0)}
    assert match_final_speakers(audio, named, embed) == {"Speaker 1": "Dana", "Speaker 2": "Sam"}
    assert match_final_speakers(audio, {"Dana": unit(0, 0, 1)}, embed) == {}  # nobody close enough


def test_names_from_the_call_land_in_the_transcript_and_suppress_the_prompt(tmp_path):
    from test_pipeline import make_meeting
    from transcript_assistant.speakers import should_prompt

    folder = make_meeting(tmp_path)
    write_json_atomic(folder / "live" / "voices.json", {"voices": {
        "Voice 1": {"centroid": A.tolist(), "count": 3, "first_seen_s": 0, "name": "Dana Lee"}}})
    turns = [Turn(0.0, 1.5, "S0")]
    meta = transcribe_meeting(folder, WordModel(), "fake", diarizer=lambda a: turns, embedder=lambda a: A)
    assert meta["speaker_names"]["Speaker 1"] == "Dana Lee"
    assert meta["speaker_names_source"].startswith("named during the call")
    assert "**Dana Lee**" in (folder / "transcript.md").read_text(encoding="utf-8")
    assert not should_prompt(folder)


# --- session wiring -----------------------------------------------------------

class FakeRec:
    def __init__(self, audio_dir, on_others_audio=None):
        self.feed = on_others_audio

    def start(self):
        if self.feed:
            for _ in range(24):
                self.feed(tone(300, 0.5))

    def stop(self):
        return 12.0, [ChannelStats("others", frames=12 * CAPTURE_RATE, peak=0.3),
                      ChannelStats("mic", frames=12 * CAPTURE_RATE, peak=0.3)]


def test_session_runs_live_naming_and_reports_it(tmp_path):
    opened = []
    factory = lambda folder: LiveSpeakers(folder, freq_embed, lambda f, v: opened.append(v))  # noqa: E731
    session = start_session(tmp_path, "Call", recording_factory=FakeRec, live_factory=factory)
    meta = finish_session(session)
    assert opened == ["Voice 1"]
    assert meta["live_voices"] == {"heard": 1, "named": 0}


def test_live_factory_failure_does_not_block_recording(tmp_path):
    def boom(folder):
        raise ImportError("pyannote not installed")

    session = start_session(tmp_path, "Call", recording_factory=FakeRec, live_factory=boom)
    meta = finish_session(session)
    assert meta["status"] == "recorded"
    assert any("live speaker naming unavailable" in w for w in meta["warnings"])
