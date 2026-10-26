"""Track the questions asked during a call: which are answered, and which one is current.

Everything here is plain rules on the live transcript, run on this machine; no
text is sent anywhere. The rules:

* A sentence is a question when Whisper ends it with "?", or when it opens like
  one ("can you", "do we", "is there", ...). One- and two-word tags ("right?",
  "you know?") are not.
* The current question is the newest one still open. When it is answered, the
  newest earlier open question (from the last 10 minutes) becomes current again.
* The current question counts as answered when someone other than the asker says
  enough that isn't itself a question: a handful of words, or a reply that opens
  with yes/no. Answers that defer ("I'll get back to you", "not sure") mark it
  "follow-up" instead, which still counts as not answered.
* When the far end is one shared label ("Others", with speaker separation off),
  a reply from that same label counts only after a pause and with more words,
  since it might be the asker carrying on.
* Whatever is said in reply to the newest question, for up to 40 seconds after
  it, is kept with it as the exchange (`Question.replies`), so the notes show how
  it was answered and not only the first words that counted. A deferral anywhere
  in the exchange turns an automatic "answered" into "follow-up".

These are guesses, so the question window lets you correct any of them; your
marks always win (see `effective_status`).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

OPEN, ANSWERED, FOLLOW_UP, DISMISSED = "open", "answered", "follow-up", "dismissed"

CURRENT_WINDOW_S = 600.0   # an older open question doesn't become current again by itself
ANSWER_WORDS = 5           # words of non-question reply that count as an answer
GROUP_ANSWER_WORDS = 10    # ...when the asker and the replier share one label
GROUP_PAUSE_S = 1.5        # ...and only after this long a pause
REPEAT_WINDOW_S = 60.0     # the same person re-asking within this window is the same question
EXCHANGE_WINDOW_S = 40.0   # replies this long after a question are kept with it
MAX_REPLIES = 6            # ...up to this many

_END = re.compile(r"[.?!]+")
_ABBREV = re.compile(r"\b(?:dr|mr|mrs|ms|st|vs|etc|jr|sr|inc|no|e\.g|i\.e|a\.m|p\.m)\.$", re.I)
_WORD = re.compile(r"[A-Za-z0-9']+")
_TAGS = {"right", "okay", "ok", "yeah", "yes", "no", "huh", "really", "sorry", "what", "hmm", "you know",
         "is it", "isn't it", "correct", "eh", "pardon", "hello", "see"}
_OPENERS = re.compile(
    r"^(?:so |and |but |okay,? |um,? |uh,? )*"
    r"(?:(?:can|could|would|will|should|shall|do|does|did|is|are|was|were|have|has|had|may|might|"
    r"am|isn't|aren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't)"
    r" (?:you|we|i|they|he|she|it|anyone|anybody|someone|there|that|this|these|those|y'all)\b"
    r"|any (?:questions|thoughts|concerns|objections|updates)\b"
    r"|(?:what|why|how|when|where|who|whom|whose|which)\b(?! (?:i|we) (?:mean|said|need|want|think)\b))",
    re.I)
_YES_NO = re.compile(r"^(?:yes|yeah|yep|yup|no|nope|nah|sure|absolutely|definitely|correct|exactly|"
                     r"of course|not really|not yet|i do|i did|we do|we did|it is|it does|that's right)\b", re.I)
_DEFER = re.compile(r"\b(?:i don't know|i'm not sure|not sure|no idea|get back to you|circle back|find out|"
                    r"look into (?:it|that|this)|check on (?:it|that|this)|follow up on|take that offline|"
                    r"let me check|i'll check|need to check|good question,? i)\b", re.I)
_FILLER = {"um", "uh", "hmm", "mm", "so", "okay", "ok", "yeah", "right", "well", "oh", "good", "question",
           "great", "that's", "a", "interesting", "let", "me", "think", "see"}


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sentence_spans(text: str) -> list[tuple[int, str]]:
    """(offset, sentence) pairs. A period inside a number ("3.5") or after "Dr." doesn't end one."""
    spans, start = [], 0

    def take(end: int) -> None:
        nonlocal start
        raw = text[start:end]
        s = raw.strip()
        if s and words(s):
            spans.append((start + len(raw) - len(raw.lstrip()), s))
        start = end

    for m in _END.finditer(text):
        after = text[m.end():m.end() + 1]
        if after and not after.isspace():
            continue  # "3.5", "e.g.x": not the end of a sentence
        if m.group() == "." and _ABBREV.search(text[start:m.end()].rstrip()):
            continue
        take(m.end())
    take(len(text))
    return spans


def sentences(text: str) -> list[str]:
    return [s for _, s in sentence_spans(text)]


def is_question(sentence: str) -> bool:
    s = sentence.strip()
    w = words(s)
    if not w:
        return False
    if len(w) <= 2 and " ".join(w) in _TAGS:
        return False
    if s.endswith("?"):
        return True
    return len(w) >= 4 and bool(_OPENERS.match(s)) and not s.endswith("!")


@dataclass
class Question:
    qid: str
    t: float               # seconds into the call
    asker: str
    text: str
    status: str = OPEN
    answered_by: str = ""
    answer: str = ""       # the first words of the reply, so you can see what counted
    answered_t: float | None = None
    end_t: float = 0.0     # when the utterance holding the question ended
    replies: list[dict] = field(default_factory=list)  # {"t", "speaker", "text", "ref"}: the exchange
    ref: object = None     # whatever the caller passed to `observe` for the utterance that asked
    reply_words: int = field(default=0, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("reply_words")
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Question":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def effective_status(q: Question, marks: dict[str, str]) -> str:
    return marks.get(q.qid) or q.status


def pick_current(questions: list[Question], marks: dict[str, str], now_t: float) -> Question | None:
    """The newest open question.

    The last question asked on the call stays current until it is answered or
    dismissed, however long ago: nobody has asked anything since, so it is still
    the one on the table. An earlier question only becomes current again while
    it is from the last 10 minutes.
    """
    open_qs = [q for q in questions if effective_status(q, marks) == OPEN]
    if not open_qs:
        return None
    newest = max(open_qs, key=lambda q: q.t)
    if newest is max(questions, key=lambda q: q.t):
        return newest
    return newest if now_t - newest.t <= CURRENT_WINDOW_S else None


@dataclass
class QuestionTracker:
    group_labels: set[str] = field(default_factory=lambda: {"Others"})
    questions: list[Question] = field(default_factory=list)
    marks: dict[str, str] = field(default_factory=dict)  # from the question window; always wins
    now_t: float = 0.0

    def current(self) -> Question | None:
        return pick_current(self.questions, self.marks, self.now_t)

    def observe(self, t: float, end_t: float, speaker: str, text: str, ref: object = None) -> list[Question]:
        """Feed one utterance, in time order. Returns the questions it created or changed.

        `ref` is kept on any question this utterance asks and on any reply it
        adds to an exchange, so a caller can find the utterance again.
        """
        self.now_t = max(self.now_t, end_t)
        spans = sentence_spans(text)
        sents = [s for _, s in spans]
        first_q = next((i for i, s in enumerate(sents) if is_question(s)), len(sents))
        changed = []
        reply = text[:spans[first_q][0]].strip() if first_q < len(spans) else text.strip()
        if reply:
            q = self._answer(t, speaker, reply)
            if q is not None:
                changed.append(q)
            q = self._extend_exchange(t, speaker, reply, ref)
            if q is not None and q not in changed:
                changed.append(q)
        for s in sents[first_q:]:
            if is_question(s):
                q = self._ask(t, end_t, speaker, s, ref)
                if q is not None:
                    changed.append(q)
        return changed

    def _ask(self, t: float, end_t: float, speaker: str, text: str, ref: object = None) -> Question | None:
        for q in reversed(self.questions):
            if t - q.t > REPEAT_WINDOW_S:
                break
            if q.asker == speaker and SequenceMatcher(None, q.text.lower(), text.lower()).ratio() >= 0.75:
                q.end_t = end_t  # asked again, or rephrased: still the same question
                return None
        q = Question(f"q{len(self.questions) + 1}", round(t, 1), speaker, text, end_t=round(end_t, 1), ref=ref)
        self.questions.append(q)
        return q

    def _extend_exchange(self, t: float, speaker: str, reply: str, ref: object) -> Question | None:
        """Keep what is said after the newest question with it, whoever answered it."""
        if not self.questions:
            return None
        q = self.questions[-1]
        if t - q.end_t > EXCHANGE_WINDOW_S or len(q.replies) >= MAX_REPLIES:
            return None
        if speaker == q.asker and (speaker not in self.group_labels or t - q.end_t < GROUP_PAUSE_S):
            return None
        if q.status != OPEN and q.answered_by and speaker != q.answered_by:
            return None  # once answered, only the answerer carrying on still belongs to the exchange
        q.replies.append({"t": round(t, 1), "speaker": speaker, "text": " ".join(reply.split()), "ref": ref})
        if q.status == ANSWERED and _DEFER.search(reply):
            q.status = FOLLOW_UP  # "yes, but let me check" is a follow-up, not an answer
        return q

    def _answer(self, t: float, speaker: str, reply: str) -> Question | None:
        q = self.current()
        if q is None:
            return None
        same_label = speaker == q.asker
        if same_label and (speaker not in self.group_labels or t - q.end_t < GROUP_PAUSE_S):
            return None
        if _DEFER.search(reply):
            return self._close(q, FOLLOW_UP, t, speaker, reply)
        content = [w for w in words(reply) if w not in _FILLER]
        q.reply_words += len(content)
        needed = GROUP_ANSWER_WORDS if same_label else ANSWER_WORDS
        if q.reply_words >= needed or (not same_label and _YES_NO.match(reply.strip())):
            return self._close(q, ANSWERED, t, speaker, reply)
        return None

    @staticmethod
    def _close(q: Question, status: str, t: float, speaker: str, reply: str) -> Question:
        q.status, q.answered_by, q.answered_t = status, speaker, round(t, 1)
        q.answer = " ".join(reply.split()[:25])
        return q

    def to_json(self) -> dict:
        cur = self.current()
        return {"now_t": round(self.now_t, 1), "current": cur.qid if cur else None,
                "questions": [q.to_json() for q in self.questions]}


def load_questions(data: dict) -> list[Question]:
    return [Question.from_json(q) for q in data.get("questions", [])]


def render_questions_markdown(title: str, questions: list[Question], marks: dict[str, str],
                              names: dict[str, str] | None = None) -> str:
    """questions.md for a finished call: open ones first, since those are what you still owe."""
    from .merge import fmt_ts

    names = names or {}
    who = lambda s: names.get(s, s)  # noqa: E731
    groups = {OPEN: [], FOLLOW_UP: [], ANSWERED: []}
    for q in questions:
        groups.get(effective_status(q, marks), []).append(q)
    lines = [f"# Questions: {title}", "",
             "Detected automatically from the live transcript; statuses are best guesses unless you marked them.",
             ""]
    for status, heading in ((OPEN, "Not answered"), (FOLLOW_UP, "Needs follow-up"), (ANSWERED, "Answered")):
        lines += [f"## {heading} ({len(groups[status])})", ""]
        for q in groups[status]:
            mark = " (marked by you)" if q.qid in marks else ""
            lines.append(f"- [{fmt_ts(q.t)}] **{who(q.asker)}**: {q.text}{mark}")
            if status != OPEN and q.answered_by:
                lines.append(f"  - {who(q.answered_by)}: {q.answer}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
