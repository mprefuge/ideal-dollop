"""Post-call notes: named questions with exchanges, carried marks, action items, meeting.md."""

from pathlib import Path

from transcript_assistant.merge import Segment
from transcript_assistant.notes import (ActionItem, build_notes, carry_marks, find_action_items,
                                        render_meeting_markdown, track_questions, write_meeting_notes)
from transcript_assistant.questions import (ANSWERED, FOLLOW_UP, OPEN, Question, QuestionTracker, pick_current,
                                            sentences)
from transcript_assistant.speakers import apply_names
from transcript_assistant.store import read_json, write_json_atomic

NAMES = {"Me": "Micah", "Speaker 1": "Sam Ortiz", "Speaker 2": "Dana Lee"}


def meeting_segments() -> list[Segment]:
    return [
        Segment(0, 3, "Morning everyone. Can everyone hear me?", "mic"),
        Segment(3.5, 5, "Yes, loud and clear.", "others", "Speaker 1"),
        Segment(6, 10, "Great. What's the go-live date for the Stripe sync?", "others", "Speaker 2"),
        Segment(11, 15, "We're targeting the second week of November. Dr. Lee signed off on 3.5 percent.", "mic"),
        Segment(16, 19, "Okay, and who owns the reconciliation report?", "others", "Speaker 2"),
        Segment(20, 24, "I'm not sure, I'll have to get back to you on that.", "mic"),
        Segment(25, 29, "Dana, can you send me the contract by Friday?", "others", "Speaker 1"),
        Segment(30, 32, "Sure, I'll send it this afternoon.", "others", "Speaker 2"),
        Segment(33, 37, "We need to schedule the board review before the end of the month.", "mic"),
        Segment(38, 40, "Let's see, I'll be honest, that's tight.", "others", "Speaker 1"),
    ]


# --- tracker changes ---------------------------------------------------------------

def test_sentences_keep_abbreviations_and_decimals_together():
    assert sentences("Dr. Lee signed off on 3.5 percent. Can you confirm?") == \
        ["Dr. Lee signed off on 3.5 percent.", "Can you confirm?"]


def test_exchange_keeps_the_replies_with_the_question():
    t = QuestionTracker()
    t.observe(0, 3, "Dana Lee", "What's the go-live date?", ref=0)
    t.observe(4, 6, "Micah", "Second week of November, assuming legal signs.", ref=1)
    t.observe(7, 9, "Micah", "I'll confirm the exact day tomorrow.", ref=2)
    t.observe(10, 12, "Dana Lee", "Perfect, thanks.", ref=3)  # the asker: not part of the answer
    (q,) = t.questions
    assert q.status == ANSWERED and q.ref == 0
    assert [r["ref"] for r in q.replies] == [1, 2]
    assert q.replies[0]["speaker"] == "Micah"


def test_a_deferral_later_in_the_exchange_turns_answered_into_follow_up():
    t = QuestionTracker()
    t.observe(0, 3, "Dana Lee", "How many refunds were there in September?")
    t.observe(4, 6, "Micah", "Somewhere around forty, I believe, maybe more.")
    assert t.questions[0].status == ANSWERED
    t.observe(7, 9, "Micah", "Actually let me check and get back to you with the real number.")
    assert t.questions[0].status == FOLLOW_UP


def test_someone_else_changing_the_subject_is_not_part_of_the_exchange():
    t = QuestionTracker()
    t.observe(0, 3, "Sam Ortiz", "Can you send me the contract?")
    t.observe(4, 6, "Dana Lee", "Sure, I'll send it this afternoon.")
    t.observe(7, 10, "Micah", "We need to schedule the board review before month end.")
    assert [r["speaker"] for r in t.questions[0].replies] == ["Dana Lee"]


def test_the_last_question_asked_stays_current_however_old():
    t = QuestionTracker()
    t.observe(0, 2, "Others", "When does the contract renew?")
    t.now_t = 700
    assert t.current().qid == "q1"
    t.observe(701, 705, "Me", "It renews at the end of the fiscal year, in June.")
    assert t.questions[0].status == ANSWERED


# --- questions on the final transcript ----------------------------------------

def test_final_questions_carry_speaker_names():
    qs = track_questions(meeting_segments(), NAMES)
    assert [q.asker for q in qs] == ["Micah", "Dana Lee", "Dana Lee", "Sam Ortiz"]
    by_text = {q.text: q for q in qs}
    stripe = by_text["What's the go-live date for the Stripe sync?"]
    assert stripe.status == ANSWERED and stripe.answered_by == "Micah"
    assert by_text["Okay, and who owns the reconciliation report?"].status == FOLLOW_UP


def test_unseparated_far_end_still_uses_the_group_rule():
    segs = [Segment(0, 3, "Who owns the reconciliation report?", "others"),
            Segment(3.5, 6, "Finance has it now, I think, maybe.", "others")]
    (q,) = track_questions(segs, {"Others": "Everyone else"})
    assert q.asker == "Everyone else" and q.status == OPEN


def test_live_marks_carry_over_by_time_and_wording():
    live = [Question("q1", 0.2, "Me", "Can everyone hear me?"),
            Question("q2", 6.1, "Others", "What's the go live date for the stripe sink?"),
            Question("q3", 300.0, "Others", "Something the final transcript never heard?")]
    final = track_questions(meeting_segments(), NAMES)
    marks, unmatched = carry_marks(live, {"q1": "dismissed", "q2": "follow-up", "q3": "answered"}, final)
    by_text = {q.text: q.qid for q in final}
    assert marks == {by_text["Can everyone hear me?"]: "dismissed",
                     by_text["What's the go-live date for the Stripe sync?"]: "follow-up"}
    assert [u["mark"] for u in unmatched] == ["answered"]


# --- action items -----------------------------------------------------------------

def owners(items: list[ActionItem]) -> list[tuple[str, str]]:
    return [(a.owner, a.kind) for a in items]


def test_action_items_find_commitments_requests_group_and_follow_ups():
    segs = meeting_segments()
    qs = track_questions(segs, NAMES)
    items = find_action_items(segs, NAMES, qs, {})
    assert owners(items) == [("Micah", "follow-up"), ("Dana Lee", "request"), ("Team", "group")]
    follow_up, request, group = items
    assert "who owns the reconciliation report" in follow_up.text
    assert request.due == "Friday"  # "sure, I'll send it this afternoon" accepted the request, not a second item
    assert group.due == "before the end of the month"


def test_request_owner_is_the_only_other_person_when_nobody_is_named():
    segs = [Segment(0, 3, "Could you send me the deck after this?", "others", "Speaker 1"),
            Segment(4, 6, "Will do.", "mic")]
    items = find_action_items(segs, NAMES, [], {})
    assert owners(items) == [("Micah", "request")]


def test_small_talk_is_not_an_action_item():
    segs = [Segment(0, 2, "Can you hear me okay?", "mic"),
            Segment(3, 5, "Let's see what the numbers say.", "others", "Speaker 1"),
            Segment(6, 8, "I'll be honest, it's tight.", "others", "Speaker 1"),
            Segment(9, 11, "Should we ship on Friday?", "others", "Speaker 2"),
            Segment(12, 14, "I'll check.", "mic")]  # too short to be a task
    assert find_action_items(segs, NAMES, [], {}) == []


def test_your_marks_change_which_questions_become_follow_ups():
    segs = meeting_segments()
    qs = track_questions(segs, NAMES)
    stripe = next(q for q in qs if "Stripe" in q.text)
    items = find_action_items(segs, NAMES, qs, {stripe.qid: FOLLOW_UP})
    assert [a.kind for a in items].count("follow-up") == 2


# --- meeting.md ---------------------------------------------------------------------

def test_meeting_markdown_has_every_section_and_markers(tmp_path):
    segs = meeting_segments()
    notes = build_notes(tmp_path, segs, NAMES)
    md = render_meeting_markdown("Weekly sync", "2026-10-26", segs, NAMES, [], notes)
    for heading in ("## Participants", "## Transcript", "## Questions", "## Action items"):
        assert heading in md
    assert "- **Micah**: spoke for" in md
    assert "[Q2] Great. What's the go-live date" in md
    assert "[A2] We're targeting" in md
    assert "- [Q2] [00:00:06] **Dana Lee**: What's the go-live date" in md
    assert "  - **Micah**: We're targeting the second week of November" in md
    assert "- [ ] Micah: Get back to Dana Lee on: Okay, and who owns the reconciliation report? [00:00:20]" in md
    assert "- [ ] Dana Lee: Dana, can you send me the contract by Friday? (due Friday) [00:00:25]" in md
    assert "Speaker 2" not in md


def test_write_meeting_notes_applies_live_marks_and_reports_counts(tmp_path):
    (tmp_path / "live").mkdir()
    write_json_atomic(tmp_path / "live" / "questions.json", {"questions": [
        {"qid": "q1", "t": 0.2, "asker": "Me", "text": "Can everyone hear me?", "status": "answered"}]})
    write_json_atomic(tmp_path / "live" / "question_marks.json", {"q1": "dismissed"})
    meta = {"title": "Weekly sync", "speaker_names": NAMES}
    counts = write_meeting_notes(tmp_path, meta, meeting_segments(), [])
    md = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "Can everyone hear me?" in md  # still in the transcript...
    assert "[Q1]" not in md  # ...but no longer listed as a question
    assert counts["questions"] == {"asked": 3, "answered": 2, "follow-up": 1}
    assert counts["action_items"] == 3


def test_renaming_a_speaker_regenerates_the_notes(tmp_path):
    folder = tmp_path
    (folder / "audio").mkdir()
    meta = {"title": "Weekly sync", "started_at": "2026-10-26", "speaker_names": {"Me": "Micah"}}
    write_json_atomic(folder / "meta.json", meta)
    write_json_atomic(folder / "transcript.json", {"segments": [s.to_dict(meta["speaker_names"])
                                                                for s in meeting_segments()]})
    apply_names(folder, {"Speaker 2": "Dana Lee"})
    md = (folder / "meeting.md").read_text(encoding="utf-8")
    assert "**Dana Lee** [00:00:06]" in md and "Speaker 1" in md
    assert "- [ ] Dana Lee: Dana, can you send me the contract" in md
    assert read_json(folder / "meta.json")["notes"]["action_items"] == 3


def test_notes_failure_never_loses_the_transcript(tmp_path, monkeypatch):
    from transcript_assistant import notes
    from transcript_assistant.transcribe import write_transcript

    def boom(*a, **k):
        raise RuntimeError("no notes today")

    monkeypatch.setattr(notes, "write_meeting_notes", boom)
    meta = {"title": "Weekly sync", "speaker_names": {}}
    write_transcript(tmp_path, meta, meeting_segments(), [])
    assert (tmp_path / "transcript.md").exists()
    assert meta["notes"] == {"error": "RuntimeError: no notes today"}


def test_pick_current_dismissed_last_question_falls_back_to_window():
    qs = [Question("q1", 0, "Me", "First?"), Question("q2", 5, "Me", "Second?")]
    assert pick_current(qs, {"q2": "dismissed"}, 8).qid == "q1"
    assert pick_current(qs, {"q2": "dismissed"}, 700) is None
