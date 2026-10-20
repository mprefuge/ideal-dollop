"""Transcribe both channels while the call is going, and feed the question tracker.

Each channel is cut into utterances at pauses (or every 15 seconds of
continuous talk), and each utterance is transcribed with a small Whisper model
on the GPU. This is only for the live view: the full transcript is still made
after the call with the large model, from the recorded audio.

Files in the meeting folder:
    live/transcript.jsonl      one line per utterance: {"t", "end", "speaker", "text"}
    live/questions.json        tracker state and status line (owned by the transcriber only)
    live/question_marks.json   {"q3": "answered", ...} written by the question window only
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .audio import CAPTURE_RATE, WHISPER_RATE
from .live import _resample_48k_to_16k
from .merge import Segment, _is_echo
from .questions import QuestionTracker, load_questions, render_questions_markdown
from .store import read_json, write_json_atomic

log = logging.getLogger(__name__)

FRAME = 480                 # 30 ms at 16 kHz
PRE_ROLL_FRAMES = 10        # keep 0.3 s before speech starts, so first words aren't clipped
END_SILENCE_S = 0.7         # a pause this long ends an utterance
MIN_SPEECH_S = 0.4          # shorter bursts are clicks and coughs
MAX_UTTERANCE_S = 15.0      # long monologues are cut so text keeps appearing
MIN_LEVEL = 0.006           # RMS below this is never speech
COMMIT_LAG_S = 10.0         # give up waiting for the other channel after this long
HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "."}

LABELS = {"mic": "Me", "others": "Others"}


class Segmenter:
    """Energy-based utterance cutter for one 16 kHz channel. Pure; no threads."""

    def __init__(self):
        self.t = 0.0                      # seconds consumed
        self.noise = 0.002
        self._pre: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
        self._cur: list[np.ndarray] = []
        self._start = 0.0
        self._speech = 0
        self._silence = 0
        self._rest = np.zeros(0, dtype=np.float32)

    @property
    def open_since(self) -> float | None:
        return self._start if self._cur else None

    def feed(self, audio16k: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """Returns finished utterances as (start seconds, audio)."""
        buf = np.concatenate([self._rest, np.asarray(audio16k, dtype=np.float32)])
        n = buf.size // FRAME
        self._rest = buf[n * FRAME:]
        out = []
        for i in range(n):
            frame = buf[i * FRAME:(i + 1) * FRAME]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            speech = rms >= max(MIN_LEVEL, 3 * self.noise)
            # Track the noise floor quickly in pauses and slowly during speech, so a steady hum
            # loud enough to look like speech stops looking like it.
            k = 0.0002 if speech else 0.05
            self.noise = (1 - k) * self.noise + k * rms
            self._step(frame, speech, out)
            self.t += FRAME / WHISPER_RATE
        return out

    def _step(self, frame: np.ndarray, speech: bool, out: list) -> None:
        if not self._cur:
            if speech:
                self._start = self.t - len(self._pre) * FRAME / WHISPER_RATE
                self._cur = list(self._pre) + [frame]
                self._speech, self._silence = 1, 0
            else:
                self._pre.append(frame)
            return
        self._cur.append(frame)
        if speech:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        length = len(self._cur) * FRAME / WHISPER_RATE
        if self._silence * FRAME / WHISPER_RATE >= END_SILENCE_S or length >= MAX_UTTERANCE_S:
            self._emit(out)

    def _emit(self, out: list) -> None:
        if self._speech * FRAME / WHISPER_RATE >= MIN_SPEECH_S:
            out.append((self._start, np.concatenate(self._cur)))
        self._cur, self._speech, self._silence = [], 0, 0
        self._pre.clear()

    def flush(self) -> list[tuple[float, np.ndarray]]:
        out = []
        if self._cur:
            self._emit(out)
        return out

    def skip(self, seconds: float) -> list[tuple[float, np.ndarray]]:
        """Audio was dropped: close what's open and move the clock on."""
        out = self.flush()
        self.t += seconds
        return out


def transcribe_utterance(model, audio: np.ndarray, language: str | None) -> str:
    segments, _info = model.transcribe(audio, language=language, vad_filter=True, beam_size=1,
                                       condition_on_previous_text=False)
    parts = [s.text.strip() for s in segments if getattr(s, "no_speech_prob", 0.0) < 0.6]
    text = " ".join(p for p in parts if p)
    return "" if text.lower() in HALLUCINATIONS else text


class LiveTranscriber:
    """Runs in a background thread for one recording. `load_model()` is called in that thread."""

    def __init__(self, folder: Path, load_model: Callable[[], object], language: str | None = None,
                 my_name: str = "Me", on_start: Callable[[Path], object] | None = None):
        self.folder = folder
        self.dir = folder / "live"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.load_model = load_model
        self.language = language
        self.names = {"Me": my_name} if my_name and my_name != "Me" else {}
        self.on_start = on_start
        self.tracker = QuestionTracker()
        self.status = "starting"
        self.errors = 0
        self.lines = 0
        self._model = None
        self._q: queue.Queue = queue.Queue(maxsize=1200)  # about 5 minutes of both channels
        self._skipped = {"mic": 0, "others": 0}  # 48 kHz frames dropped while the queue was full
        self._skip_lock = threading.Lock()
        self._seg = {"mic": Segmenter(), "others": Segmenter()}
        self._pending: list[Segment] = []
        self._recent: deque[Segment] = deque(maxlen=40)
        self._marks_mtime = None
        self._thread = threading.Thread(target=self._run, name="live-transcribe", daemon=True)

    # Called from the capture threads: must never block them.
    def feed_mic(self, block48k: np.ndarray) -> None:
        self._put("mic", block48k)

    def feed_others(self, block48k: np.ndarray) -> None:
        self._put("others", block48k)

    def _put(self, channel: str, block: np.ndarray) -> None:
        try:
            self._q.put_nowait((channel, np.asarray(block, dtype=np.float32)))
        except queue.Full:
            with self._skip_lock:
                self._skipped[channel] += len(block)

    def start(self) -> None:
        self._write_state()
        self._thread.start()
        if self.on_start is not None:
            try:
                self.on_start(self.folder)
            except Exception:
                log.exception("could not open the question window")

    def stop(self, timeout: float = 60.0) -> None:
        self._q.put(None)
        self._thread.join(timeout)

    # --- worker ---------------------------------------------------------------

    def _run(self) -> None:
        self.status = "loading speech model"
        self._write_state()
        try:
            self._model = self.load_model()
            self.status = "listening"
        except Exception as exc:
            self.status = f"live transcription unavailable: {type(exc).__name__}: {exc}"
            log.exception("live transcription model failed to load")
        self._write_state()
        while True:
            item = self._q.get()
            if item is None:
                break
            channel, block = item
            with self._skip_lock:
                dropped, self._skipped[channel] = self._skipped[channel], 0
            utts = self._seg[channel].skip(dropped / CAPTURE_RATE) if dropped else []
            utts += self._seg[channel].feed(_resample_48k_to_16k(block))
            self._handle(channel, utts)
        for channel, seg in self._seg.items():
            self._handle(channel, seg.flush())
        self._commit(final=True)
        if self._model is not None:
            self.status = "call ended"
        self._write_state()

    def _handle(self, channel: str, utts: list[tuple[float, np.ndarray]]) -> None:
        for start, audio in utts:
            if self._model is None:
                continue
            try:
                text = transcribe_utterance(self._model, audio, self.language)
            except Exception:
                self.errors += 1
                log.exception("live transcription failed")
                continue
            if text:
                self._pending.append(Segment(round(start, 2), round(start + audio.size / WHISPER_RATE, 2),
                                             text, channel))
        self._commit()

    def _commit(self, final: bool = False) -> None:
        """Pass utterances on in time order, once the other channel can no longer overlap them.

        Waiting lets a mic utterance that is only the speakers bleeding into the
        microphone be recognised as echo before it reaches the tracker.
        """
        latest = max(s.t for s in self._seg.values())
        ready = []
        for s in sorted(self._pending, key=lambda s: s.start):
            other = self._seg["others" if s.channel == "mic" else "mic"]
            other_done = other.t >= s.end and (other.open_since is None or other.open_since > s.end)
            if final or other_done or latest - s.end > COMMIT_LAG_S:
                ready.append(s)
        if not ready:
            self._absorb_marks()
            return
        others = [o for o in list(self._recent) + self._pending if o.channel == "others"]
        taken = {id(s) for s in ready}
        self._pending = [s for s in self._pending if id(s) not in taken]
        with open(self.dir / "transcript.jsonl", "a", encoding="utf-8") as f:
            for s in ready:
                if s.channel == "mic" and any(_is_echo(s, o, 0.6) for o in others):
                    continue
                self._recent.append(s)
                speaker = LABELS[s.channel]
                f.write(json.dumps({"t": s.start, "end": s.end, "speaker": speaker, "text": s.text}) + "\n")
                self.lines += 1
                self.tracker.observe(s.start, s.end, speaker, s.text)
        self._absorb_marks()
        self._write_state()

    def _absorb_marks(self) -> None:
        path = self.dir / "question_marks.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self._marks_mtime:
            return
        try:
            self.tracker.marks = dict(read_json(path))
            self._marks_mtime = mtime
            self._write_state()
        except (OSError, ValueError):
            pass

    def _write_state(self) -> None:
        state = self.tracker.to_json()
        state.update(status=self.status, lines=self.lines, names=self.names)
        write_json_atomic(self.dir / "questions.json", state)

    def write_summary(self) -> dict:
        """questions.md in the meeting folder, with your marks applied. Returns counts for meta.json."""
        return write_questions_summary(self.folder, self.names)


def read_marks(folder: Path) -> dict[str, str]:
    path = folder / "live" / "question_marks.json"
    try:
        return dict(read_json(path))
    except (OSError, ValueError):
        return {}


def write_mark(folder: Path, qid: str, status: str | None) -> dict[str, str]:
    """Set (or with None, clear) your own status for one question. Only the question window calls this."""
    marks = read_marks(folder)
    if status is None:
        marks.pop(qid, None)
    else:
        marks[qid] = status
    write_json_atomic(folder / "live" / "question_marks.json", marks)
    return marks


def write_questions_summary(folder: Path, names: dict[str, str] | None = None) -> dict:
    from .questions import effective_status

    path = folder / "live" / "questions.json"
    if not path.exists():
        return {}
    state = read_json(path)
    questions = load_questions(state)
    if not questions:
        return {"asked": 0}
    marks = read_marks(folder)
    names = {**state.get("names", {}), **(names or {})}
    title = read_json(folder / "meta.json").get("title", folder.name) if (folder / "meta.json").exists() \
        else folder.name
    (folder / "questions.md").write_text(render_questions_markdown(title, questions, marks, names),
                                         encoding="utf-8")
    counts: dict[str, int] = {"asked": 0}
    for q in questions:
        st = effective_status(q, marks)
        if st != "dismissed":
            counts["asked"] += 1
            counts[st] = counts.get(st, 0) + 1
    return counts


def launch_question_window(root: Path, folder: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "transcript_assistant.cli", "--root", str(root),
                             "questions", str(folder)])
