from types import SimpleNamespace

import numpy as np
import pytest

from transcript_assistant.audio import CAPTURE_RATE, WHISPER_RATE
from transcript_assistant.live_transcribe import (LiveTranscriber, Segmenter, read_marks, transcribe_utterance,
                                                  write_mark, write_questions_summary)
from transcript_assistant.questions import (ANSWERED, DISMISSED, FOLLOW_UP, OPEN, QuestionTracker, is_question,
                                            load_questions, pick_current, sentences)
from transcript_assistant.recorder import ChannelStats
from transcript_assistant.session import finish_session, start_session
from transcript_assistant.store import read_json


# --- what counts as a question ----------------------------------------------------

@pytest.mark.parametrize("text", [
    "What's the timeline for the integration?",
    "Can you send the agenda over",
    "so do we have a contract yet.",
    "Any questions on the budget",
    "We're shipping Friday, right?",
])
def test_questions(text):
    assert is_question(text)


@pytest.mark.parametrize("text", [
    "Right?", "You know?", "Okay?", "We shipped it on Friday.", "What we need is a contract.",
    "How great!", "",
])
def test_not_questions(text):
    assert not is_question(text)


def test_sentences_split_on_punctuation():
    assert sentences("Yes. It's Tuesday. Can you send it?") == ["Yes.", "It's Tuesday.", "Can you send it?"]


# --- answered, unanswered, current ---------------------------------------------

def test_other_person_answering_closes_the_question():
    t = QuestionTracker()
    t.observe(0, 3, "Others", "What's the go-live date for the Stripe sync?")
    assert t.current().text.startswith("What's the go-live")
    t.observe(4, 8, "Me", "We're targeting the second week of November.")
    q = t.questions[0]
    assert q.status == ANSWERED and q.answered_by == "Me" and q.answer.startswith("We're targeting")
    assert t.current() is None


def test_short_yes_no_counts_but_filler_does_not():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "Did you get the invoice?")
    t.observe(3, 4, "Me", "Hmm, good question.")
    assert t.questions[0].status == OPEN
    t.observe(5, 6, "Me", "Yes.")
    assert t.questions[0].status == ANSWERED


def test_asker_carrying_on_is_not_an_answer():
    t = QuestionTracker()
    t.observe(0, 3, "Me", "Can you share the report? I need it for the board meeting on Thursday.")
    t.observe(4, 8, "Me", "Because the board wants numbers before they vote on it.")
    assert t.questions[0].status == OPEN


def test_shared_far_end_label_needs_a_pause_and_more_words():
    t = QuestionTracker()
    t.observe(0, 3, "Others", "Who owns the reconciliation report?")
    t.observe(3.5, 6, "Others", "Finance has it now, I think, maybe.")  # right after: probably the asker
    assert t.questions[0].status == OPEN
    t.observe(9, 14, "Others", "Finance owns it now, Dana runs it every Monday morning before standup.")
    assert t.questions[0].status == ANSWERED


def test_deferral_marks_follow_up():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "How many refunds were there in September?")
    t.observe(3, 5, "Me", "I'll have to get back to you on that.")
    assert t.questions[0].status == FOLLOW_UP
    assert t.current() is None


def test_earlier_open_question_becomes_current_again():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "When does the contract renew?")
    t.observe(3, 5, "Others", "And who signs it?")
    assert t.current().qid == "q2"
    t.observe(6, 9, "Me", "The provost signs every contract over ten thousand.")
    assert t.questions[1].status == ANSWERED
    assert t.current().qid == "q1"
    t.now_t = 700
    assert t.current() is None  # too old to be the current question by itself any more
    assert t.questions[0].status == OPEN  # but it is still listed as not answered


def test_answer_then_new_question_in_one_breath():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "Is the API key rotated?")
    t.observe(3, 7, "Me", "Yes, last Friday. Can you confirm the webhook still fires?")
    assert [q.status for q in t.questions] == [ANSWERED, OPEN]
    assert t.current().asker == "Me"


def test_repeating_a_question_does_not_duplicate_it():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "What's the budget for this?")
    t.observe(10, 12, "Others", "So what is the budget for this?")
    assert len(t.questions) == 1


def test_your_marks_win():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "Who has the keys?")
    t.marks = {"q1": DISMISSED}
    assert t.current() is None
    qs = load_questions(t.to_json())
    assert pick_current(qs, {}, 5).qid == "q1"


# --- utterance cutting --------------------------------------------------------

def tone16(seconds, amp=0.2, freq=300):
    n = np.arange(int(seconds * WHISPER_RATE)) / WHISPER_RATE
    return (amp * np.sin(2 * np.pi * freq * n)).astype(np.float32)


def silence16(seconds):
    return np.zeros(int(seconds * WHISPER_RATE), dtype=np.float32)


def test_segmenter_cuts_at_pauses_and_keeps_wall_clock():
    seg = Segmenter()
    out = seg.feed(np.concatenate([silence16(1), tone16(2), silence16(1), tone16(0.1), silence16(1), tone16(1.5),
                                   silence16(1)]))
    assert len(out) == 2  # the 0.1 s click is dropped
    (s1, a1), (s2, a2) = out
    assert s1 == pytest.approx(0.7, abs=0.05)  # 0.3 s of pre-roll before speech at 1.0 s
    assert s2 == pytest.approx(4.8, abs=0.05)
    assert a1.size / WHISPER_RATE == pytest.approx(3.0, abs=0.1)  # pre-roll + speech + closing pause


def test_segmenter_cuts_long_monologues():
    seg = Segmenter()
    out = seg.feed(tone16(40)) + seg.flush()
    assert [round(a.size / WHISPER_RATE) for _, a in out] == [15, 15, 10]


def test_hallucinated_thank_you_is_dropped():
    class M:
        def transcribe(self, audio, **kw):
            return iter([SimpleNamespace(text=" Thank you.", no_speech_prob=0.1)]), None

    assert transcribe_utterance(M(), silence16(1), None) == ""


# --- live transcriber end to end ---------------------------------------------------

class ScriptModel:
    """Fake Whisper: returns the next scripted line for each utterance it's given."""

    def __init__(self, lines):
        self.lines = list(lines)

    def transcribe(self, audio, **kw):
        text = self.lines.pop(0) if self.lines else ""
        return iter([SimpleNamespace(text=" " + text, no_speech_prob=0.05)]), None


def tone48(seconds, amp=0.2):
    n = np.arange(int(seconds * CAPTURE_RATE)) / CAPTURE_RATE
    return (amp * np.sin(2 * np.pi * 300 * n)).astype(np.float32)


def silence48(seconds):
    return np.zeros(int(seconds * CAPTURE_RATE), dtype=np.float32)


def feed(lt, channel, audio):
    f = lt.feed_mic if channel == "mic" else lt.feed_others
    for i in range(0, audio.size, CAPTURE_RATE // 2):
        f(audio[i:i + CAPTURE_RATE // 2])


def test_live_transcriber_tracks_questions_across_channels(tmp_path):
    model = ScriptModel(["Can you walk me through the refund flow?", "Sure, refunds post as negative charges."])
    lt = LiveTranscriber(tmp_path, lambda: model, my_name="Micah")
    lt.start()
    # Others asks during 0-2 s; Micah answers during 3-5 s. Both channels run on the same clock.
    feed(lt, "others", np.concatenate([tone48(2), silence48(4)]))
    feed(lt, "mic", np.concatenate([silence48(3), tone48(2), silence48(1)]))
    lt.stop()
    state = read_json(tmp_path / "live" / "questions.json")
    assert state["status"] == "call ended"
    (q,) = load_questions(state)
    assert q.asker == "Others" and q.status == ANSWERED and q.answered_by == "Me"
    lines = (tmp_path / "live" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and '"speaker": "Others"' in lines[0]
    assert state["names"] == {"Me": "Micah"}


def test_mic_echo_of_the_far_end_is_ignored(tmp_path):
    model = ScriptModel(["Is the webhook live?", "Is the webhook live?"])
    lt = LiveTranscriber(tmp_path, lambda: model)
    lt.start()
    feed(lt, "others", np.concatenate([tone48(2), silence48(2)]))
    feed(lt, "mic", np.concatenate([tone48(2, amp=0.05), silence48(2)]))  # the speakers, heard by the mic
    lt.stop()
    (q,) = load_questions(read_json(tmp_path / "live" / "questions.json"))
    assert q.asker == "Others" and q.status == OPEN
    assert len((tmp_path / "live" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_model_failure_is_reported_not_fatal(tmp_path):
    def broken():
        raise RuntimeError("CUDA out of memory")

    lt = LiveTranscriber(tmp_path, broken)
    lt.start()
    feed(lt, "others", tone48(3))
    lt.stop()
    assert "unavailable: RuntimeError" in read_json(tmp_path / "live" / "questions.json")["status"]


def test_marks_reach_the_tracker_and_the_summary(tmp_path):
    (tmp_path / "meta.json").write_text('{"title": "Sync"}', encoding="utf-8")
    model = ScriptModel(["What's the go-live date?", "Who signs the contract?"])
    lt = LiveTranscriber(tmp_path, lambda: model)
    lt.start()
    feed(lt, "others", np.concatenate([tone48(2), silence48(2), tone48(2), silence48(2)]))
    lt.stop()
    write_mark(tmp_path, "q1", ANSWERED)
    assert read_marks(tmp_path) == {"q1": ANSWERED}
    counts = write_questions_summary(tmp_path)
    assert counts == {"asked": 2, ANSWERED: 1, OPEN: 1}
    md = (tmp_path / "questions.md").read_text(encoding="utf-8")
    assert md.index("## Not answered (1)") < md.index("Who signs the contract?") < md.index("## Answered (1)")
    assert "What's the go-live date? (marked by you)" in md


# --- session wiring -----------------------------------------------------------------

class FakeRec:
    def __init__(self, audio_dir, on_others_audio=None, on_mic_audio=None):
        self.others, self.mic = on_others_audio, on_mic_audio

    def start(self):
        for block in np.split(np.concatenate([tone48(2), silence48(2)]), 8):
            self.others(block)
            self.mic(np.zeros_like(block))

    def stop(self):
        return 4.0, [ChannelStats("others", frames=4 * CAPTURE_RATE, peak=0.2),
                     ChannelStats("mic", frames=4 * CAPTURE_RATE, peak=0.2)]


def test_session_runs_live_transcription_and_saves_questions(tmp_path):
    opened = []
    factory = lambda folder: LiveTranscriber(folder, lambda: ScriptModel(["Any blockers this week?"]),  # noqa: E731
                                             on_start=opened.append)
    session = start_session(tmp_path, "Standup", recording_factory=FakeRec, transcriber_factory=factory)
    meta = finish_session(session)
    assert opened == [session.folder]
    assert meta["live_questions"] == {"asked": 1, OPEN: 1}
    assert "Any blockers this week?" in (session.folder / "questions.md").read_text(encoding="utf-8")


def test_both_live_helpers_share_the_far_end_audio(tmp_path):
    from test_live import freq_embed
    from transcript_assistant.live import LiveSpeakers

    session = start_session(tmp_path, "Call", recording_factory=FakeRec,
                            live_factory=lambda f: LiveSpeakers(f, freq_embed),
                            transcriber_factory=lambda f: LiveTranscriber(f, lambda: ScriptModel(["Hi there."])))
    finish_session(session)
    assert session.live._t > 0  # live naming saw the audio too
    assert session.transcriber.lines == 1


# --- staying alive on Windows -------------------------------------------------------

def test_atomic_write_retries_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    import os

    from transcript_assistant import store

    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError(13, "Access is denied")  # what Windows raises while the window reads it
        real(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    store.write_json_atomic(tmp_path / "questions.json", {"ok": True})
    assert read_json(tmp_path / "questions.json") == {"ok": True} and len(calls) == 3
    assert not list(tmp_path.glob(".*.tmp"))


def test_a_failing_step_does_not_end_live_transcription(tmp_path):
    model = ScriptModel(["What's the go-live date?", "Who signs the contract?"])
    lt = LiveTranscriber(tmp_path, lambda: model)
    real_commit, failed = lt._commit, []

    def commit_once_broken(final=False):
        if not failed:
            failed.append(1)
            raise PermissionError(13, "Access is denied")
        real_commit(final)

    lt._commit = commit_once_broken
    lt.start()
    feed(lt, "others", np.concatenate([tone48(2), silence48(2), tone48(2), silence48(2)]))
    lt.stop()
    state = read_json(tmp_path / "live" / "questions.json")
    assert state["status"] == "call ended"
    assert [q.text for q in load_questions(state)] == ["What's the go-live date?", "Who signs the contract?"]
    assert lt.errors == 1 and "PermissionError" in lt.last_error


def test_meeting_is_still_marked_recorded_when_live_questions_fail(tmp_path):
    class Broken(LiveTranscriber):
        def write_summary(self):
            raise PermissionError(13, "Access is denied")

    factory = lambda folder: Broken(folder, lambda: ScriptModel(["Any blockers?"]))  # noqa: E731
    session = start_session(tmp_path, "Standup", recording_factory=FakeRec, transcriber_factory=factory)
    meta = finish_session(session)
    assert read_json(session.folder / "meta.json")["status"] == "recorded"
    assert any("could not save live questions" in w for w in meta["warnings"])
