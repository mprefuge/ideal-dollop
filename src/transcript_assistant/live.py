"""Spot new far-end voices during a call so they can be named while the meeting is live.

Every few seconds of far-end audio is turned into a speaker embedding (a voice
fingerprint) and compared with the voices heard so far. A voice that matches
none of them for several windows in a row, while matching itself, is new: its
clip is saved and a small naming window pops up. Answers come back as files,
so the window runs in its own process and never blocks recording.

Files in the meeting folder:
    live/voices.json           tracker state (owned by the tracker only)
    live/voice_N.wav           a few seconds of that voice, for Play
    live/answers/voice_N.json  {"name": "..."} written by the naming window
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np

from .audio import CAPTURE_RATE, WHISPER_RATE
from .store import read_json, write_json_atomic

log = logging.getLogger(__name__)

WINDOW_S = 3.0
MIN_RMS = 0.01          # quieter windows are silence or noise, not speech
CONFIRM_WINDOWS = 3     # unmatched windows needed before announcing a new voice
CONFIRM_SPAN_S = 30.0   # ...and they must fall within this many seconds


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


@dataclass
class Voice:
    voice_id: str
    centroid: np.ndarray
    count: int
    first_seen_s: float
    name: str = ""

    def add(self, emb: np.ndarray) -> None:
        self.centroid = (self.centroid * self.count + emb) / (self.count + 1)
        self.count += 1

    def to_json(self) -> dict:
        return {"centroid": [round(float(x), 6) for x in self.centroid], "count": self.count,
                "first_seen_s": round(self.first_seen_s, 1), "name": self.name}


@dataclass
class VoiceTracker:
    """Pure clustering logic: embeddings in, "new voice" events out."""

    threshold: float = 0.55
    voices: dict[str, Voice] = field(default_factory=dict)
    _pending: list[tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)  # (t, emb, audio)
    last_clip: np.ndarray | None = None  # audio of the most recently confirmed new voice

    def observe(self, t: float, emb: np.ndarray, audio: np.ndarray) -> tuple[str | None, Voice | None]:
        """Returns (matched voice id or None, newly confirmed Voice or None)."""
        best_id, best_sim = None, -1.0
        for v in self.voices.values():
            sim = cosine(emb, v.centroid)
            if sim > best_sim:
                best_id, best_sim = v.voice_id, sim
        if best_id is not None and best_sim >= self.threshold:
            self.voices[best_id].add(emb)
            return best_id, None

        # Unmatched: keep only recent pending windows that sound like this one.
        self._pending = [(pt, pe, pa) for pt, pe, pa in self._pending
                         if t - pt <= CONFIRM_SPAN_S and cosine(pe, emb) >= self.threshold]
        self._pending.append((t, emb, audio))
        if len(self._pending) < CONFIRM_WINDOWS:
            return None, None
        embs = np.stack([pe for _, pe, _ in self._pending])
        vid = f"Voice {len(self.voices) + 1}"
        voice = Voice(vid, embs.mean(axis=0), len(embs), self._pending[0][0])
        self.voices[vid] = voice
        self.last_clip = np.concatenate([pa for _, _, pa in self._pending])
        self._pending = []
        return vid, voice

    def set_name(self, voice_id: str, name: str) -> None:
        """Name a voice. Naming it the same as another voice merges the two."""
        name = name.strip()
        voice = self.voices.get(voice_id)
        if voice is None:
            return
        twin = next((v for v in self.voices.values() if v.voice_id != voice_id and name and v.name == name), None)
        if twin is not None:
            total = twin.count + voice.count
            twin.centroid = (twin.centroid * twin.count + voice.centroid * voice.count) / total
            twin.count = total
            voice.name = name
            voice.centroid = twin.centroid  # keep the id so earlier references still resolve
        else:
            voice.name = name

    def named_centroids(self) -> dict[str, np.ndarray]:
        return {v.name: v.centroid for v in self.voices.values() if v.name}

    def to_json(self) -> dict:
        return {"threshold": self.threshold, "voices": {k: v.to_json() for k, v in self.voices.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "VoiceTracker":
        t = cls(data.get("threshold", 0.55))
        for vid, v in data.get("voices", {}).items():
            t.voices[vid] = Voice(vid, np.asarray(v["centroid"], dtype=np.float32), v["count"],
                                  v["first_seen_s"], v.get("name", ""))
        return t


def _resample_48k_to_16k(block: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    g = gcd(CAPTURE_RATE, WHISPER_RATE)
    return resample_poly(block, WHISPER_RATE // g, CAPTURE_RATE // g).astype(np.float32)


class LiveSpeakers:
    """Runs a VoiceTracker on the far-end audio of one recording, in a background thread."""

    def __init__(self, folder: Path, embed: Callable[[np.ndarray], np.ndarray],
                 on_new_voice: Callable[[Path, str], object] | None = None, threshold: float = 0.55):
        self.folder = folder
        self.dir = folder / "live"
        (self.dir / "answers").mkdir(parents=True, exist_ok=True)
        self.embed = embed
        self.on_new_voice = on_new_voice
        self.tracker = VoiceTracker(threshold)
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=600)
        self._buf = np.zeros(0, dtype=np.float32)
        self._t = 0.0  # seconds of far-end audio consumed
        self._thread = threading.Thread(target=self._run, name="live-speakers", daemon=True)
        self.errors = 0

    # Called from the capture thread: must never block it.
    def feed(self, block48k: np.ndarray) -> None:
        try:
            self._q.put_nowait(np.asarray(block48k, dtype=np.float32))
        except queue.Full:
            pass  # falling behind only delays naming; the recording itself is unaffected

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._q.put(None)
        self._thread.join(timeout)
        self._collect_answers()
        self._save()

    def _run(self) -> None:
        while True:
            block = self._q.get()
            if block is None:
                return
            self._buf = np.concatenate([self._buf, _resample_48k_to_16k(block)])
            win = int(WINDOW_S * WHISPER_RATE)
            while self._buf.size >= win:
                chunk, self._buf = self._buf[:win], self._buf[win:]
                self.process_window(chunk)
            self._collect_answers()

    def process_window(self, chunk: np.ndarray) -> None:
        t = self._t
        self._t += chunk.size / WHISPER_RATE
        if float(np.sqrt(np.mean(chunk ** 2))) < MIN_RMS:
            return
        try:
            emb = np.asarray(self.embed(chunk), dtype=np.float32).reshape(-1)
        except Exception:
            self.errors += 1
            log.exception("speaker embedding failed")
            return
        _, new = self.tracker.observe(t, emb, chunk)
        if new is not None:
            self._announce(new)

    def _announce(self, voice: Voice) -> None:
        import soundfile as sf

        sf.write(self.dir / f"{_slug(voice.voice_id)}.wav", self.tracker.last_clip, WHISPER_RATE)
        self._save()
        log.info("New voice %s at %.0fs", voice.voice_id, voice.first_seen_s)
        if self.on_new_voice is not None:
            try:
                self.on_new_voice(self.folder, voice.voice_id)
            except Exception:
                log.exception("could not open naming window")

    def _collect_answers(self) -> None:
        changed = False
        for path in sorted((self.dir / "answers").glob("*.json")):
            try:
                answer = json.loads(path.read_text(encoding="utf-8"))
                self.tracker.set_name(answer["voice_id"], answer.get("name", ""))
                path.unlink()
                changed = True
            except (OSError, ValueError, KeyError):
                continue
        if changed:
            self._save()

    def _save(self) -> None:
        write_json_atomic(self.dir / "voices.json", self.tracker.to_json())


def _slug(voice_id: str) -> str:
    return voice_id.lower().replace(" ", "_")


def write_answer(folder: Path, voice_id: str, name: str) -> None:
    write_json_atomic(folder / "live" / "answers" / f"{_slug(voice_id)}.json", {"voice_id": voice_id, "name": name})


def absorb_late_answers(folder: Path) -> None:
    """Apply names typed into a naming window after the call had already ended."""
    path = folder / "live" / "voices.json"
    answers = sorted((folder / "live" / "answers").glob("*.json"))
    if not path.exists() or not answers:
        return
    tracker = VoiceTracker.from_json(read_json(path))
    for a in answers:
        try:
            ans = json.loads(a.read_text(encoding="utf-8"))
            tracker.set_name(ans["voice_id"], ans.get("name", ""))
            a.unlink()
        except (OSError, ValueError, KeyError):
            continue
    write_json_atomic(path, tracker.to_json())


def live_names(folder: Path) -> dict[str, np.ndarray]:
    """Named voices from a finished call: name -> centroid."""
    absorb_late_answers(folder)
    path = folder / "live" / "voices.json"
    if not path.exists():
        return {}
    state = read_json(path)
    out: dict[str, list[np.ndarray]] = {}
    for v in state.get("voices", {}).values():
        if v.get("name"):
            out.setdefault(v["name"], []).append(np.asarray(v["centroid"], dtype=np.float32))
    return {name: np.mean(cs, axis=0) for name, cs in out.items()}


def match_final_speakers(speaker_audio: dict[str, np.ndarray], named: dict[str, np.ndarray],
                         embed: Callable[[np.ndarray], np.ndarray], threshold: float = 0.55) -> dict[str, str]:
    """Map final diarized speakers ("Speaker 2") to names given during the call."""
    out = {}
    if not named:
        return out
    for sid, audio in speaker_audio.items():
        if audio.size < WHISPER_RATE:  # under a second: too little to fingerprint
            continue
        emb = np.asarray(embed(audio), dtype=np.float32).reshape(-1)
        name, sim = max(((n, cosine(emb, c)) for n, c in named.items()), key=lambda x: x[1])
        if sim >= threshold:
            out[sid] = name
    return out


def launch_voice_dialog(root: Path, folder: Path, voice_id: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "transcript_assistant.cli", "--root", str(root),
                             "name-voice", str(folder), voice_id])


def load_embedder(token: str | None = None, device: str = "cuda") -> Callable[[np.ndarray], np.ndarray]:
    """pyannote's speaker-embedding model (the one its diarization pipeline uses)."""
    import os

    import torch
    from pyannote.audio import Inference, Model

    token = token or os.environ.get("HF_TOKEN")
    name = "pyannote/wespeaker-voxceleb-resnet34-LM"
    try:
        model = Model.from_pretrained(name, token=token)
    except TypeError:  # pyannote.audio 3.x
        model = Model.from_pretrained(name, use_auth_token=token)
    inference = Inference(model, window="whole")
    inference.to(torch.device(device))

    def embed(audio16k: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32)).unsqueeze(0)
        return np.asarray(inference({"waveform": wav, "sample_rate": WHISPER_RATE})).reshape(-1)

    return embed
