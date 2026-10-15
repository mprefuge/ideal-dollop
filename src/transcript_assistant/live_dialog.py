"""The small mid-call window: "New voice on the call. Who is this?" """

from __future__ import annotations

import threading
from pathlib import Path

from .live import _slug, write_answer
from .speakers import name_suggestions
from .store import read_json


def _play(path: Path) -> None:
    def run():
        try:
            import soundcard as sc
            import soundfile as sf

            data, rate = sf.read(path, dtype="float32")
            sc.default_speaker().play(data, samplerate=rate)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def run_voice_dialog(root: Path, folder: Path, voice_id: str, my_emails: list[str]) -> str:
    """Returns the name entered, or "" if skipped."""
    import tkinter as tk
    from tkinter import ttk

    meta = read_json(folder / "meta.json")
    voices = {}
    vpath = folder / "live" / "voices.json"
    if vpath.exists():
        voices = read_json(vpath).get("voices", {})
    in_call = sorted({v["name"] for v in voices.values() if v.get("name")})
    suggestions = in_call + [n for n in name_suggestions(root, folder, my_emails) if n not in in_call]
    clip = folder / "live" / f"{_slug(voice_id)}.wav"

    win = tk.Tk()
    win.title("New voice on the call")
    win.attributes("-topmost", True)
    win.resizable(False, False)
    result = {"name": ""}

    body = ttk.Frame(win, padding=12)
    body.grid()
    ttk.Label(body, text=f"Someone new is speaking in \"{meta.get('title', folder.name)}\". Who is it?",
              wraplength=380).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    box = ttk.Combobox(body, values=suggestions, width=30)
    box.grid(row=1, column=0, columnspan=2, sticky="w")
    box.focus_set()
    if clip.exists():
        ttk.Button(body, text="Play", width=6, command=lambda: _play(clip)).grid(row=1, column=2, padx=(8, 0))
    if in_call:
        ttk.Label(body, text="Picking a name already used on this call means it's the same person.",
                  foreground="#555", wraplength=380).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def save(_e=None):
        result["name"] = box.get().strip()
        if result["name"]:
            write_answer(folder, voice_id, result["name"])
        win.destroy()

    buttons = ttk.Frame(body)
    buttons.grid(row=3, column=0, columnspan=3, sticky="e", pady=(10, 0))
    ttk.Button(buttons, text="Skip", command=win.destroy).grid(row=0, column=0, padx=4)
    ttk.Button(buttons, text="Save", command=save).grid(row=0, column=1, padx=4)
    win.bind("<Return>", save)
    win.bind("<Escape>", lambda _e: win.destroy())
    win.mainloop()
    return result["name"]
