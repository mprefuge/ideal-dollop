"""System tray icon for `watch`: status, stop, pause, question window, open folder, quit."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from .watcher import RECORDING, Watcher


def _icon_image(recording: bool):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=(200, 30, 30, 255) if recording else (110, 110, 110, 255))
    return img


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opens Explorer on the user's own folder


def run_tray(watcher: Watcher, root: Path, stop_event: threading.Event) -> None:
    import pystray

    def is_recording(_item=None):
        return watcher.state == RECORDING

    def show_questions():
        session = watcher.session
        if session is not None and (session.folder / "live" / "questions.json").exists():
            from .live_transcribe import launch_question_window

            launch_question_window(root, session.folder)

    def transcribe_missed(icon, _item):
        n = watcher.transcribe_missed() if hasattr(watcher, "transcribe_missed") else 0
        icon.notify(f"Transcribing {n} missed meeting(s)" if n else "No missed meetings: everything is transcribed",
                    "Transcript Assistant")

    def quit_(icon, _item):
        stop_event.set()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem(lambda _item: watcher.status, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Stop recording this meeting", lambda: watcher.stop_now(), enabled=is_recording),
        pystray.MenuItem("Show questions window", show_questions, enabled=is_recording),
        pystray.MenuItem("Pause auto-record for 1 hour", lambda: watcher.pause(3600)),
        pystray.MenuItem("Resume auto-record", lambda: watcher.resume()),
        pystray.MenuItem("Transcribe missed meetings", transcribe_missed),
        pystray.MenuItem("Open transcripts folder", lambda: _open_folder(root)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_),
    )
    icon = pystray.Icon("transcript-assistant", _icon_image(False), "Transcript Assistant", menu)

    def notify(msg: str) -> None:
        try:
            icon.icon = _icon_image(is_recording())
            icon.title = f"Transcript Assistant: {watcher.status}"
            icon.notify(msg, "Transcript Assistant")
        except Exception:
            pass

    watcher.notify = notify
    loop = threading.Thread(target=watcher.run, args=(stop_event,), name="watcher", daemon=True)
    loop.start()
    icon.run()  # blocks until Quit
    stop_event.set()
    loop.join(timeout=30)
