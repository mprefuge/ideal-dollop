from types import SimpleNamespace

import numpy as np
import pytest

from transcript_assistant import cli
from transcript_assistant.diarize import Turn, assign_speakers, calendar_names
from transcript_assistant.merge import Segment
from transcript_assistant.store import read_json, write_json_atomic
from transcript_assistant.transcribe import transcribe_meeting

from test_pipeline import make_meeting


def W(start, end, text):
    return (start, end, text)


# --- assigning speakers -------------------------------------------------------

def test_segment_is_split_where_the_speaker_changes():
    seg = Segment(0.0, 4.0, "Hi Dana. Hey Sam.", "others",
                  words=[W(0.0, 0.5, " Hi"), W(0.5, 1.0, " Dana."), W(2.0, 2.5, " Hey"), W(2.5, 3.0, " Sam.")])
    turns = [Turn(0.0, 1.2, "SPEAKER_07"), Turn(1.8, 3.5, "SPEAKER_02")]
    out = assign_speakers([seg], turns)
    assert [(s.speaker, s.text) for s in out] == [("Speaker 1", "Hi Dana."), ("Speaker 2", "Hey Sam.")]
    assert (out[1].start, out[1].end) == (2.0, 3.0)


def test_numbering_follows_first_appearance_and_is_stable():
    segs = [Segment(0, 1, "a", "others"), Segment(2, 3, "b", "others"), Segment(4, 5, "c", "others")]
    turns = [Turn(0, 1, "SPEAKER_01"), Turn(2, 3, "SPEAKER_00"), Turn(4, 5, "SPEAKER_01")]
    assert [s.speaker for s in assign_speakers(segs, turns)] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_word_in_a_gap_takes_the_nearest_turn_or_the_previous_speaker():
    seg = Segment(0, 6, "one two three", "others",
                  words=[W(0.0, 0.5, " one"), W(1.6, 1.8, " two"), W(5.0, 5.2, " three")])
    turns = [Turn(0.0, 1.0, "A")]
    out = assign_speakers([seg], turns)
    # "two" is 0.7s from the turn (nearest); "three" is far, so it stays with the run it follows.
    assert [(s.speaker, s.text) for s in out] == [("Speaker 1", "one two three")]


def test_no_turns_leaves_segments_unchanged():
    segs = [Segment(0, 1, "a", "others")]
    assert assign_speakers(segs, [])[0].speaker == "Others"


# --- calendar naming ----------------------------------------------------------

EVENT_1ON1 = {"organizer": "Micah <micah@example.org>", "attendees": ["Dana Lee <dana@example.edu>"]}


def test_one_other_attendee_and_one_voice_is_named():
    segs = [Segment(0, 1, "hi", "others"), Segment(1, 2, "hello", "mic")]
    names, source = calendar_names(segs, EVENT_1ON1, ["MICAH@example.org"])
    assert names == {"Others": "Dana Lee"} and source


def test_no_guessing_when_ambiguous():
    segs = [Segment(0, 1, "hi", "others")]
    # my address unknown, so two "others" -> no name
    assert calendar_names(segs, EVENT_1ON1, []) == ({}, "")
    # two far-end voices -> no name even with one invitee
    two = [Segment(0, 1, "a", "others", "Speaker 1"), Segment(1, 2, "b", "others", "Speaker 2")]
    assert calendar_names(two, EVENT_1ON1, ["micah@example.org"]) == ({}, "")
    assert calendar_names(segs, None, ["micah@example.org"]) == ({}, "")


# --- end to end ---------------------------------------------------------------

class WordModel:
    """Fake Whisper that returns word timings when asked for them."""

    def transcribe(self, audio, **kwargs):
        loud = float(np.max(np.abs(audio))) > 0.2
        if not loud:
            return iter([SimpleNamespace(start=0.0, end=0.4, text=" Sounds good.", words=None)]), None
        words = [SimpleNamespace(start=0.0, end=0.4, word=" Budget"), SimpleNamespace(start=0.4, end=0.8, word=" is"),
                 SimpleNamespace(start=1.0, end=1.4, word=" Agreed.")]
        return iter([SimpleNamespace(start=0.0, end=1.4, text=" Budget is Agreed.",
                                     words=words if kwargs["word_timestamps"] else None)]), None


def test_transcribe_with_diarizer_labels_far_end_and_applies_my_name(tmp_path):
    folder = make_meeting(tmp_path)
    turns = [Turn(0.0, 0.9, "S0"), Turn(0.9, 1.5, "S1")]
    meta = transcribe_meeting(folder, WordModel(), "fake", diarizer=lambda audio: turns, my_name="Micah")
    assert meta["diarized"] is True
    assert meta["speakers"] == ["Me", "Speaker 1", "Speaker 2"]
    md = (folder / "transcript.md").read_text(encoding="utf-8")
    assert "**Micah**" in md and "**Speaker 2**" in md and "**Me**" not in md
    segs = read_json(folder / "transcript.json")["segments"]
    assert {s["speaker_id"] for s in segs} == {"Me", "Speaker 1", "Speaker 2"}


def test_diarizer_failure_keeps_transcript_and_warns(tmp_path):
    folder = make_meeting(tmp_path)

    def broken(audio):
        raise RuntimeError("CUDA out of memory")

    meta = transcribe_meeting(folder, WordModel(), "fake", diarizer=broken)
    assert meta["diarized"] is False
    assert any("speaker separation failed" in w for w in meta["warnings"])
    assert "Others" in meta["speakers"]


# --- rename -------------------------------------------------------------------

def test_rename_updates_transcript_and_survives_retranscription(tmp_path, capsys):
    folder = make_meeting(tmp_path)
    turns = [Turn(0.0, 0.9, "S0"), Turn(0.9, 1.5, "S1")]
    transcribe_meeting(folder, WordModel(), "fake", diarizer=lambda a: turns)
    assert cli.main(["--root", str(tmp_path), "rename", str(folder), "Speaker 2=Dana Lee"]) == 0
    assert "**Dana Lee**" in (folder / "transcript.md").read_text(encoding="utf-8")
    transcribe_meeting(folder, WordModel(), "fake", diarizer=lambda a: turns)
    assert "**Dana Lee**" in (folder / "transcript.md").read_text(encoding="utf-8")
    assert cli.main(["--root", str(tmp_path), "rename", str(folder), "Speaker 2="]) == 0
    assert "**Speaker 2**" in (folder / "transcript.md").read_text(encoding="utf-8")


def test_rename_rejects_unknown_speaker(tmp_path, capsys):
    folder = make_meeting(tmp_path)
    transcribe_meeting(folder, WordModel(), "fake")
    assert cli.main(["--root", str(tmp_path), "rename", str(folder), "Speaker 9=X"]) == 1
    assert "Speakers in this meeting: Me, Others" in capsys.readouterr().err


def test_rename_before_transcription_fails_cleanly(tmp_path):
    folder = make_meeting(tmp_path)
    assert cli.main(["--root", str(tmp_path), "rename", str(folder), "Others=X"]) == 1
