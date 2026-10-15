from pathlib import Path

import numpy as np
import pytest

from transcript_assistant.audio import CAPTURE_RATE, ChunkWriter
from transcript_assistant.diarize import Turn
from transcript_assistant.speakers import (apply_names, load_clip, mark_prompted, name_suggestions,
                                           should_prompt, unnamed_speakers)
from transcript_assistant.store import read_json, write_json_atomic
from transcript_assistant.transcribe import transcribe_meeting
from transcript_assistant.watcher import TranscriptionWorker

from test_pipeline import make_meeting
from test_speakers import WordModel

TURNS = [Turn(0.0, 0.9, "S0"), Turn(0.9, 1.5, "S1")]


def diarized_meeting(root, title="Weekly sync", event=None):
    folder = make_meeting(root, title)
    if event:
        meta = read_json(folder / "meta.json")
        meta["calendar_event"] = event
        write_json_atomic(folder / "meta.json", meta)
    transcribe_meeting(folder, WordModel(), "fake", diarizer=lambda a: TURNS)
    return folder


def test_unnamed_speakers_lists_far_end_only_with_quotes_and_clip(tmp_path):
    folder = diarized_meeting(tmp_path)
    prompts = unnamed_speakers(folder)
    assert [p.speaker_id for p in prompts] == ["Speaker 1", "Speaker 2"]  # Me is never asked about
    assert prompts[0].quotes == ["Budget is"]
    assert prompts[0].clip == (0.0, 0.8)
    assert should_prompt(folder)


def test_named_or_dismissed_meetings_do_not_prompt(tmp_path):
    folder = diarized_meeting(tmp_path)
    apply_names(folder, {"Speaker 1": "Dana Lee"})
    assert [p.speaker_id for p in unnamed_speakers(folder)] == ["Speaker 2"]
    mark_prompted(folder, "dismissed")
    assert not should_prompt(folder)


def test_undiarized_meeting_never_prompts(tmp_path):
    folder = make_meeting(tmp_path)
    transcribe_meeting(folder, WordModel(), "fake")
    assert unnamed_speakers(folder) == [] and not should_prompt(folder)


def test_suggestions_put_invitees_first_then_frequent_past_names(tmp_path):
    for _ in range(2):
        past = diarized_meeting(tmp_path, "past")
        apply_names(past, {"Speaker 1": "Sam Ortiz"})
    past = diarized_meeting(tmp_path, "past")
    apply_names(past, {"Speaker 1": "Pat Kim"})
    event = {"organizer": "Micah <micah@example.org>",
             "attendees": ["Dana Lee <dana@example.edu>", "Sam Ortiz <sam@example.edu>"]}
    folder = diarized_meeting(tmp_path, "now", event)
    assert name_suggestions(tmp_path, folder, ["micah@example.org"]) == ["Dana Lee", "Sam Ortiz", "Pat Kim"]


def test_same_name_twice_merges_turns_in_transcript(tmp_path):
    folder = diarized_meeting(tmp_path)
    apply_names(folder, {"Speaker 1": "Dana Lee", "Speaker 2": "Dana Lee"})
    md = (folder / "transcript.md").read_text(encoding="utf-8")
    assert md.count("**Dana Lee**") == 1  # one turn, not two back to back
    assert {s["speaker_id"] for s in read_json(folder / "transcript.json")["segments"]} >= {"Speaker 1", "Speaker 2"}


def test_apply_names_rejects_unknown_ids(tmp_path):
    folder = diarized_meeting(tmp_path)
    with pytest.raises(KeyError, match="Speaker 7"):
        apply_names(folder, {"Speaker 7": "X"})


def test_load_clip_spans_chunk_boundary_without_reading_everything(tmp_path):
    w = ChunkWriter(tmp_path, "others", chunk_seconds=1)
    ramp = (np.arange(3 * CAPTURE_RATE) / (3 * CAPTURE_RATE)).astype(np.float32) * 0.9
    w.write(ramp)
    w.close()
    data, rate = load_clip(tmp_path, 0.5, 1.5)
    assert rate == CAPTURE_RATE and data.size == CAPTURE_RATE
    assert data[0] == pytest.approx(ramp[CAPTURE_RATE // 2], abs=1e-3)
    assert data[-1] == pytest.approx(ramp[int(1.5 * CAPTURE_RATE) - 1], abs=1e-3)
    assert load_clip(tmp_path / "missing", 0, 1) is None


def test_worker_runs_after_hook_only_on_success(tmp_path):
    good = make_meeting(tmp_path, "good")
    called = []
    worker = TranscriptionWorker(tmp_path, load_model=lambda: object(),
                                 transcribe=lambda f, m: {"title": f.name} if f == good else 1 / 0,
                                 after=called.append)
    worker.start()
    worker.submit(Path(tmp_path / "missing"))
    worker.submit(good)
    worker.submit(None)
    worker.join(timeout=10)
    assert called == [good]
