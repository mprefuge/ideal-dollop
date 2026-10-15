from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from transcript_assistant import cli
from transcript_assistant.audio import CAPTURE_RATE, WHISPER_RATE, ChunkWriter, frames_to_pad, load_channel
from transcript_assistant.merge import Segment, drop_echo, merge, render_markdown
from transcript_assistant.recorder import ChannelStats, capture_warnings
from transcript_assistant.store import (audio_purge_candidates, new_meeting_dir, read_json, rebuild_index,
                                        write_json_atomic)
from transcript_assistant.transcribe import transcribe_meeting


def tone(seconds, rate=CAPTURE_RATE, amp=0.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# --- audio ---------------------------------------------------------------

def test_pad_only_when_behind_wall_clock():
    assert frames_to_pad(0, CAPTURE_RATE, elapsed_s=1.0) == 0
    assert frames_to_pad(0, CAPTURE_RATE, elapsed_s=1.2) == 0  # within tolerance
    assert frames_to_pad(0, CAPTURE_RATE, elapsed_s=3.0) == 2 * CAPTURE_RATE


def test_chunks_roll_over_and_reload_at_whisper_rate(tmp_path):
    w = ChunkWriter(tmp_path, "mic", chunk_seconds=1)
    for _ in range(5):
        w.write(tone(0.5))
    w.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mic_0000.flac", "mic_0001.flac", "mic_0002.flac"]
    assert w.frames_written == int(2.5 * CAPTURE_RATE)
    audio = load_channel(tmp_path, "mic")
    assert audio.size == int(2.5 * WHISPER_RATE)
    assert 0.25 < np.max(np.abs(audio)) < 0.35


def test_missing_channel_loads_empty(tmp_path):
    assert load_channel(tmp_path, "others").size == 0


# --- merge ---------------------------------------------------------------

def test_echo_of_others_on_mic_is_dropped_but_real_speech_kept():
    others = [Segment(10.0, 14.0, "Can everyone see my screen?", "others")]
    mic = [
        Segment(10.2, 13.9, "can everyone see my screen", "mic"),   # bleed
        Segment(14.5, 16.0, "Yes, I can see it.", "mic"),          # reply
        Segment(11.0, 12.0, "Sorry, go ahead.", "mic"),            # overlap, different words
    ]
    kept, dropped = drop_echo(mic, others)
    assert dropped == 1
    assert [s.text for s in kept] == ["Yes, I can see it.", "Sorry, go ahead."]


def test_short_overlapping_reply_is_not_mistaken_for_echo():
    others = [Segment(0, 2, "Quiet channel.", "others")]
    kept, dropped = drop_echo([Segment(0.5, 1.5, "Loud channel.", "mic")], others)
    assert dropped == 0 and len(kept) == 1


def test_merge_orders_by_time_and_labels_speakers():
    segs, _ = merge([Segment(5, 6, "second", "mic")], [Segment(1, 2, "first", "others")])
    assert [(s.speaker, s.text) for s in segs] == [("Others", "first"), ("Me", "second")]


def test_markdown_groups_turns_and_shows_warnings():
    segs = [Segment(0, 1, "Hi.", "others"), Segment(1, 2, "How are you?", "others"),
            Segment(65, 66, "Good.", "mic")]
    md = render_markdown("Sync", "2026-10-15T13:00:00+00:00", segs, ["mic: channel is silent"])
    assert md.count("**Others**") == 1
    assert "**Me** [00:01:05]" in md
    assert "Capture warnings" in md


# --- recorder warnings ---------------------------------------------------

def test_capture_warnings_flag_silence_short_capture_and_errors():
    stats = [
        ChannelStats("others", frames=60 * CAPTURE_RATE, peak=0.0),
        ChannelStats("mic", frames=10 * CAPTURE_RATE, peak=0.5, error="OSError: device removed"),
    ]
    w = capture_warnings(60.0, stats)
    assert any("others: channel is silent" in x for x in w)
    assert any("mic: captured 10s of 60s" in x for x in w)
    assert any("device removed" in x for x in w)


def test_healthy_capture_has_no_warnings():
    stats = [ChannelStats("others", frames=60 * CAPTURE_RATE, peak=0.4),
             ChannelStats("mic", frames=60 * CAPTURE_RATE, peak=0.4)]
    assert capture_warnings(60.0, stats) == []


# --- end to end with a fake model ----------------------------------------

class FakeModel:
    """Stands in for WhisperModel: returns canned segments per channel by loudness."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        assert audio.dtype == np.float32
        assert kwargs["vad_filter"] is True
        loud = float(np.max(np.abs(audio))) > 0.2
        seg = SimpleNamespace(start=0.5, end=1.5, text=" Can everyone hear me?" if loud else " Yes, loud and clear.")
        return iter([seg]), None


def make_meeting(root, title="Weekly sync", started=None, status="recorded"):
    started = started or datetime.now(timezone.utc)
    folder = new_meeting_dir(root, title, started)
    others = ChunkWriter(folder / "audio", "others")
    others.write(tone(2, amp=0.5))
    others.close()
    mic = ChunkWriter(folder / "audio", "mic")
    mic.write(tone(2, amp=0.1))
    mic.close()
    write_json_atomic(folder / "meta.json", {"title": title, "started_at": started.isoformat(),
                                             "status": status, "capture_warnings": ["x"], "warnings": ["x"]})
    return folder


def test_transcribe_meeting_writes_outputs_and_is_rerunnable(tmp_path):
    folder = make_meeting(tmp_path)
    for _ in range(2):  # re-running must not duplicate warnings
        meta = transcribe_meeting(folder, FakeModel(), "fake")
    assert meta["status"] == "transcribed"
    assert meta["warnings"] == ["x"]
    assert meta["segment_counts"] == {"mic": 1, "others": 1}
    segs = read_json(folder / "transcript.json")["segments"]
    assert {s["speaker"] for s in segs} == {"Me", "Others"}
    assert "Weekly sync" in (folder / "transcript.md").read_text(encoding="utf-8")


# --- store ---------------------------------------------------------------

def test_meeting_dirs_never_collide(tmp_path):
    t = datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc)
    a = new_meeting_dir(tmp_path, "Budget / Q4!", t)
    b = new_meeting_dir(tmp_path, "Budget / Q4!", t)
    assert a != b and a.name.endswith("budget-q4") and b.name.endswith("budget-q4-2")


def test_purge_never_touches_untranscribed_or_kept_audio(tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    done = make_meeting(tmp_path, "done", old, status="transcribed")
    make_meeting(tmp_path, "pending", old, status="recorded")
    kept = make_meeting(tmp_path, "kept", old, status="transcribed")
    meta = read_json(kept / "meta.json")
    meta["keep_audio"] = True
    write_json_atomic(kept / "meta.json", meta)
    make_meeting(tmp_path, "recent", status="transcribed")
    assert audio_purge_candidates(tmp_path, 7) == [done / "audio"]


def test_purge_cli_is_dry_run_without_yes(tmp_path, capsys):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    done = make_meeting(tmp_path, "done", old, status="transcribed")
    assert cli.main(["--root", str(tmp_path), "purge-audio"]) == 0
    assert (done / "audio").exists()
    assert "Dry run" in capsys.readouterr().out
    assert cli.main(["--root", str(tmp_path), "purge-audio", "--yes"]) == 0
    assert not (done / "audio").exists()
    assert "audio_purged_at" in read_json(done / "meta.json")


def test_index_survives_a_corrupt_meta(tmp_path):
    make_meeting(tmp_path, "good")
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "meta.json").write_text("{not json", encoding="utf-8")
    entries = rebuild_index(tmp_path)
    assert len(entries) == 2
    assert any("unreadable" in e.get("error", "") for e in entries)


def test_record_refuses_without_consent(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert cli.main(["--root", str(tmp_path), "record"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_transcribe_requires_target():
    with pytest.raises(SystemExit):
        cli.main(["transcribe"])


def test_cuda_dll_dirs_registered_only_on_windows(tmp_path, monkeypatch):
    import os
    import site
    import sys

    from transcript_assistant.transcribe import add_cuda_dll_dirs

    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(tmp_path / "none"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert add_cuda_dll_dirs() == []
    seen = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", seen.append, raising=False)
    monkeypatch.setenv("PATH", "")
    assert len(add_cuda_dll_dirs()) == 2 and len(seen) == 2
    assert "cublas" in os.environ["PATH"]
