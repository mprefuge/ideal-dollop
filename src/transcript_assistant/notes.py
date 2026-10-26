"""Post-call notes: who spoke, the questions with their exchanges, action items, and meeting.md.

Everything here runs on the final transcript (large model, speakers separated and
named), after the call. The live question window keeps running on the live text
during the call; its questions are matched to the final ones here so the marks
you made in it carry over. Plain rules only, nothing leaves the machine.

Action items are found by rule, so expect some noise:

* a first-person commitment ("I'll send the deck", "let me pull the numbers")
  belongs to the person who said it;
* a request ("can you send me the contract", "please share the link", "let me
  know if...") belongs to the person addressed: the one named in the sentence,
  else the only other person on the call, else nobody yet;
* a group commitment ("we need to decide by Friday", "let's schedule a review")
  is listed for the team;
* a question that got a deferral ("I'll get back to you") becomes an item for
  whoever deferred it.

Small talk that looks like one of these ("can you hear me", "let's see",
"I'll be honest") is skipped by a short list of phrases.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .merge import Segment, fmt_ts
from .questions import (ANSWERED, DISMISSED, FOLLOW_UP, OPEN, Question, QuestionTracker, _DEFER, effective_status,
                        load_questions, sentence_spans, words)
from .store import read_json

MARK_MATCH_S = 10.0        # a live question and a final one this close in time can be the same
MARK_MATCH_RATIO = 0.5     # ...when their words are at least this alike
DEDUPE_S = 60.0            # the same person saying the same thing again within this long is one item
ACCEPT_S = 30.0            # "sure, I'll do it" this soon after a request to that person is the same item
MIN_ITEM_WORDS = 3
REPLY_WORDS = 40           # how much of each reply the notes show

_LEAD = r"^(?:(?:so|and|but|okay|ok|yeah|yes|sure|alright|well|um|uh|also|then|oh)[,\s]+)*"
_COMMIT = re.compile(_LEAD + r"(?:i'll|i will|i can|i could|i'm going to|i am going to|i'm gonna|let me|i need to|"
                     r"i have to|i've got to|i got to|i gotta|i'd like to|i want to|i'm happy to|i plan to)\b", re.I)
_REQUEST = re.compile(_LEAD + r"(?:(?:can|could|would|will)\s+you\b|please\s+\w+|let me know\b|i need you to\b|"
                      r"you need to\b|you'll need to\b|make sure (?:you|to)\b|if you could\b|"
                      r"(?:\w+,\s+)?(?:can|could|would|will)\s+you\b)", re.I)
_GROUP = re.compile(_LEAD + r"(?:we need to|we should|we'll need to|we have to|we've got to|we'll|we will|let's|"
                    r"we ought to|we must)\b", re.I)
_DUE = re.compile(r"\b(?:(?:by|before|until|no later than|due)\s+(?:the\s+)?(?:end of (?:the )?(?:day|week|month|quarter)|"
                  r"eod|eow|eom|today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                  r"next (?:week|month|monday|tuesday|wednesday|thursday|friday)|(?:this|next) (?:call|meeting)|"
                  r"\d{1,2}(?:st|nd|rd|th)?(?: of)?(?: \w+)?|(?:january|february|march|april|may|june|july|august|"
                  r"september|october|november|december)(?: \d{1,2}(?:st|nd|rd|th)?)?)"
                  r"|today|tonight|tomorrow|this (?:week|afternoon|evening|morning)|next (?:week|month)|"
                  r"end of (?:the )?(?:day|week|month)|eod|eow|asap|right after (?:this|the call))\b", re.I)
_NOT_ITEM = (
    # conversation flow, not work
    "hear me", "see my screen", "see the screen", "see that", "see this", "see it", "repeat that", "say that again",
    "go back", "scroll", "hold on", "speak up", "mute", "unmute", "hang on", "bear with", "wait a",
    "let's see", "let's say", "let's go ", "let's move", "let's start", "let's get started", "let's get into",
    "let's jump", "let's talk", "let's look", "let's take a look", "let's hear", "let's begin", "let's continue",
    "let's keep", "let's think", "let's hope", "let's assume", "let's imagine", "let's pretend", "let's call it",
    "let's not", "let's just", "let me think", "let me see", "let me look", "let me just", "let me finish",
    "let me ask", "let me say", "let me explain", "let me show", "let me share", "let me pull up", "let me go back",
    "i'll be honest", "i'll say", "i'll tell you", "i'll let you go", "i'll see", "i'll think", "i'll bet",
    "i'll admit", "i'll take that", "i'll pass", "i'll stop", "i'll try to remember", "i'll leave it",
    "i will say", "i'll be", "i'll go ahead and share", "i'll go ahead and stop",
    "we'll see", "we'll go", "we'll get there", "we'll talk", "we'll be", "we will see", "we should be fine",
    "we need to be", "we need to keep", "we need to remember", "we need to understand", "we have to be",
    "i want to say", "i want to be", "i'd like to think", "i want to make sure i understand", "i need to think",
    "can you believe", "can you imagine", "would you say", "would you agree", "could you say", "can you tell me",
    "you need to know", "let me know if you can hear", "please go ahead", "please feel free", "please continue",
)


@dataclass
class ActionItem:
    t: float
    owner: str           # display name, "Team" for group items, "" when nobody was named
    speaker: str         # who said it
    text: str
    kind: str            # "commitment", "request", "group" or "follow-up"
    due: str = ""
    ref: object = None   # index of the transcript segment it came from

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("ref")
        return d


@dataclass
class Notes:
    questions: list[Question]
    marks: dict[str, str]
    unmatched_marks: list[dict]
    actions: list[ActionItem]
    speaking_s: dict[str, float]  # display name -> seconds, in order of first appearance

    def counts(self) -> dict:
        out: dict = {"questions": {"asked": 0}, "action_items": len(self.actions)}
        for q in self.questions:
            st = effective_status(q, self.marks)
            if st != DISMISSED:
                out["questions"]["asked"] += 1
                out["questions"][st] = out["questions"].get(st, 0) + 1
        return out


def display(seg: Segment, names: dict[str, str]) -> str:
    return names.get(seg.speaker, seg.speaker)


# --- questions on the final transcript -------------------------------------------

def track_questions(segments: list[Segment], names: dict[str, str]) -> list[Question]:
    """Run the question tracker over the final transcript, with display names as speakers."""
    group = {names.get("Others", "Others")}  # the far end when speakers were not separated
    tracker = QuestionTracker(group_labels=group)
    for i, seg in enumerate(segments):
        if seg.text.strip():
            tracker.observe(seg.start, seg.end, display(seg, names), seg.text, ref=i)
    return tracker.questions


def carry_marks(live_questions: list[Question], live_marks: dict[str, str],
                final: list[Question]) -> tuple[dict[str, str], list[dict]]:
    """Move the marks made in the live window onto the matching final questions.

    Live and final text come from different models, so a match is by time
    (within 10 s) and rough wording. Marks with no match are returned so they
    are not lost silently.
    """
    by_id = {q.qid: q for q in live_questions}
    marks, unmatched = {}, []
    for qid, status in live_marks.items():
        lq = by_id.get(qid)
        if lq is None:
            continue
        best, best_ratio = None, MARK_MATCH_RATIO
        for fq in final:
            if abs(fq.t - lq.t) > MARK_MATCH_S:
                continue
            ratio = SequenceMatcher(None, lq.text.lower(), fq.text.lower()).ratio()
            if ratio >= best_ratio:
                best, best_ratio = fq, ratio
        if best is not None and best.qid not in marks:
            marks[best.qid] = status
        else:
            unmatched.append({"t": lq.t, "asker": lq.asker, "text": lq.text, "mark": status})
    return marks, unmatched


# --- action items ------------------------------------------------------------------

def _clean(sentence: str) -> str:
    return " ".join(sentence.split()).strip()


def _addressee(sentence: str, speaker: str, participants: list[str]) -> str:
    low = sentence.lower()
    for p in participants:
        if p != speaker and p.lower() in low:
            return p
    for p in participants:  # first names
        first = p.split()[0].lower() if p.split() else ""
        if p != speaker and first and re.search(rf"\b{re.escape(first)}\b", low):
            return p
    others = [p for p in participants if p != speaker]
    return others[0] if len(others) == 1 else ""


def _classify(sentence: str) -> str | None:
    low = sentence.lower()
    if any(phrase in low for phrase in _NOT_ITEM):
        return None
    if _REQUEST.match(low):
        return "request"
    if sentence.rstrip().endswith("?"):
        return None  # "should we ship Friday?" is a question, not a commitment
    if _COMMIT.match(low):
        return "commitment"
    if _GROUP.match(low):
        return "group"
    return None


def _duplicate(item: ActionItem, items: list[ActionItem]) -> bool:
    for other in items:
        if abs(other.t - item.t) <= DEDUPE_S and other.speaker == item.speaker and \
                SequenceMatcher(None, other.text.lower(), item.text.lower()).ratio() >= 0.7:
            return True
        if item.kind == "commitment" and other.kind == "request" and other.owner == item.owner \
                and 0 <= item.t - other.t <= ACCEPT_S:
            other.due = other.due or item.due  # "sure, I'll send it tomorrow" accepts the request
            return True
    return False


def _due(sentence: str) -> str:
    m = _DUE.search(sentence)
    if not m:
        return ""
    return re.sub(r"^(?:by|due)\s+", "", m.group().strip(), flags=re.I)


def find_action_items(segments: list[Segment], names: dict[str, str], questions: list[Question],
                      marks: dict[str, str]) -> list[ActionItem]:
    participants: list[str] = []
    for seg in segments:
        who = display(seg, names)
        if who not in participants:
            participants.append(who)
    items: list[ActionItem] = []
    for q in questions:
        if effective_status(q, marks) == FOLLOW_UP:
            owner = q.answered_by or ""
            items.append(ActionItem(q.answered_t if q.answered_t is not None else q.t, owner, owner,
                                    f"Get back to {q.asker} on: {q.text}", "follow-up", ref=q.ref))
    for i, seg in enumerate(segments):
        who = display(seg, names)
        for _, sentence in sentence_spans(seg.text):
            if len(words(sentence)) < MIN_ITEM_WORDS:
                continue
            kind = _classify(sentence)
            if kind is None:
                continue
            if kind == "request":
                owner = _addressee(sentence, who, participants)
            elif kind == "group":
                owner = "Team"
            else:
                owner = who
            item = ActionItem(seg.start, owner, who, _clean(sentence), kind, _due(sentence), ref=i)
            if kind == "commitment" and _DEFER.search(sentence) and any(
                    o.kind == "follow-up" and o.speaker == who and abs(o.t - item.t) <= 5 for o in items):
                continue  # already listed as the follow-up to a question
            if not _duplicate(item, items):
                items.append(item)
    items.sort(key=lambda a: a.t)
    return items


# --- the notes file ------------------------------------------------------------------

def speaking_time(segments: list[Segment], names: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for seg in segments:
        if seg.text.strip():
            who = display(seg, names)
            out[who] = out.get(who, 0.0) + max(0.0, seg.end - seg.start)
    return out


def _fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _short(text: str, limit: int = REPLY_WORDS) -> str:
    parts = text.split()
    return " ".join(parts[:limit]) + (" ..." if len(parts) > limit else "")


def render_meeting_markdown(title: str, started_at: str, segments: list[Segment], names: dict[str, str],
                            warnings: list[str], notes: Notes) -> str:
    asked: dict[object, list[str]] = {}
    replied: dict[object, list[str]] = {}
    for q in notes.questions:
        if effective_status(q, notes.marks) == DISMISSED:
            continue
        asked.setdefault(q.ref, []).append(q.qid)
        for r in q.replies:
            replied.setdefault(r.get("ref"), []).append(q.qid)

    lines = [f"# {title}", "", f"Started: {started_at}", ""]
    if warnings:
        lines += ["> **Capture warnings**", ">"] + [f"> - {w}" for w in warnings] + [""]

    lines += ["## Participants", ""]
    for who, secs in notes.speaking_s.items():
        lines.append(f"- **{who}**: spoke for {_fmt_minutes(secs)}")
    if not notes.speaking_s:
        lines.append("- (no speech transcribed)")
    lines.append("")

    lines += ["## Transcript", "",
              "Questions are marked [Q#] where they are asked and [A#] where they are answered; "
              "the list under Questions below has the same numbers.", ""]
    prev = None
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        who = display(seg, names)
        if who != prev:
            lines += ["", f"**{who}** [{fmt_ts(seg.start)}]"]
            prev = who
        tags = [f"[{q.upper()}]" for q in asked.get(i, [])] + [f"[A{q[1:]}]" for q in replied.get(i, [])
                                                                 if q not in asked.get(i, [])]
        lines.append((" ".join(tags) + " " if tags else "") + text)
    lines.append("")

    groups: dict[str, list[Question]] = {OPEN: [], FOLLOW_UP: [], ANSWERED: []}
    for q in notes.questions:
        groups.get(effective_status(q, notes.marks), []).append(q)
    lines += ["## Questions", "",
              "Found by rule from the transcript; statuses are best guesses unless you marked them "
              "in the question window during the call.", ""]
    if not any(groups.values()):
        lines += ["No questions were detected.", ""]
    for status, heading in ((OPEN, "Not answered"), (FOLLOW_UP, "Needs follow-up"), (ANSWERED, "Answered")):
        if not groups[status]:
            continue
        lines += [f"### {heading} ({len(groups[status])})", ""]
        for q in groups[status]:
            mark = " (marked by you)" if q.qid in notes.marks else ""
            lines.append(f"- [{q.qid.upper()}] [{fmt_ts(q.t)}] **{q.asker}**: {q.text}{mark}")
            for r in q.replies:
                lines.append(f"  - **{r['speaker']}**: {_short(r['text'])}")
            if status != OPEN and not q.replies and q.answered_by:
                lines.append(f"  - **{q.answered_by}**: {_short(q.answer)}")
        lines.append("")
    if notes.unmatched_marks:
        lines += [f"{len(notes.unmatched_marks)} mark(s) from the live window did not match a question here "
                  "and were kept in meta.json.", ""]

    lines += ["## Action items", ""]
    if not notes.actions:
        lines.append("- [ ] (none detected)")
    for a in notes.actions:
        owner = a.owner or "Unassigned"
        due = f" (due {a.due})" if a.due else ""
        lines.append(f"- [ ] {owner}: {a.text}{due} [{fmt_ts(a.t)}]")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_notes(folder: Path, segments: list[Segment], names: dict[str, str]) -> Notes:
    questions = track_questions(segments, names)
    marks: dict[str, str] = {}
    unmatched: list[dict] = []
    live_state = folder / "live" / "questions.json"
    live_marks = folder / "live" / "question_marks.json"
    if live_state.exists() and live_marks.exists():
        try:
            live_qs = load_questions(read_json(live_state))
            marks, unmatched = carry_marks(live_qs, dict(read_json(live_marks)), questions)
        except (OSError, ValueError):
            pass
    actions = find_action_items(segments, names, questions, marks)
    return Notes(questions, marks, unmatched, actions, speaking_time(segments, names))


def write_meeting_notes(folder: Path, meta: dict, segments: list[Segment], warnings: list[str]) -> dict:
    """Write meeting.md from the final transcript. Returns counts and leftovers for meta.json."""
    names = meta.get("speaker_names", {})
    notes = build_notes(folder, segments, names)
    md = render_meeting_markdown(meta.get("title", folder.name), meta.get("started_at", ""), segments, names,
                                 warnings, notes)
    (folder / "meeting.md").write_text(md, encoding="utf-8")
    out = notes.counts()
    if notes.unmatched_marks:
        out["unmatched_live_marks"] = notes.unmatched_marks
    return out
