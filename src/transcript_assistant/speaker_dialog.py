"""The "who is speaking?" window shown after a meeting with unnamed speakers."""

from __future__ import annotations

import threading
from pathlib import Path

from .speakers import (SpeakerPrompt, apply_names, load_clip, mark_prompted, name_suggestions,
                       unnamed_speakers)
from .store import read_json


def _play(audio_dir: Path, prompt: SpeakerPrompt) -> None:
    def run():
        try:
            import soundcard as sc

            clip = load_clip(audio_dir, *prompt.clip)
            if clip is not None:
                data, rate = clip
                sc.default_speaker().play(data, samplerate=rate)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def run_dialog(root: Path, folder: Path, my_emails: list[str]) -> str:
    """Show the dialog; returns "saved", "later" or "dismissed"."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    prompts = unnamed_speakers(folder)
    if not prompts:
        return "saved"
    meta = read_json(folder / "meta.json")
    suggestions = name_suggestions(root, folder, my_emails)
    audio_dir = folder / "audio"
    has_audio = any(audio_dir.glob("others_*.flac"))

    win = tk.Tk()
    win.title(f"Who is speaking? {meta.get('title', folder.name)}")
    win.attributes("-topmost", True)
    win.resizable(False, False)
    outcome = {"value": "later"}

    body = ttk.Frame(win, padding=14)
    body.grid(sticky="nsew")
    ttk.Label(body, text=f"{len(prompts)} new speaker(s) in \"{meta.get('title', folder.name)}\". "
                         "Pick a name, type one, or leave blank. Giving two speakers the same name "
                         "merges them.", wraplength=520).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

    boxes: dict[str, ttk.Combobox] = {}
    row = 1
    for p in prompts:
        spoken = f"{p.seconds_spoken:.0f} s" if p.seconds_spoken < 60 else f"{p.seconds_spoken / 60:.0f} min"
        ttk.Label(body, text=f"{p.speaker_id}  ({spoken})",
                  font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w")
        box = ttk.Combobox(body, values=suggestions, width=32)
        box.grid(row=row, column=1, sticky="w", padx=8)
        boxes[p.speaker_id] = box
        if has_audio and p.clip:
            ttk.Button(body, text="Play", width=6, command=lambda p=p: _play(audio_dir, p)).grid(row=row, column=2)
        row += 1
        for q in p.quotes:
            ttk.Label(body, text=f"“{q[:140]}”", foreground="#555", wraplength=520).grid(
                row=row, column=0, columnspan=3, sticky="w", padx=(12, 0))
            row += 1
        ttk.Separator(body).grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

    buttons = ttk.Frame(body)
    buttons.grid(row=row, column=0, columnspan=3, sticky="e")

    def save():
        changes = {sid: box.get() for sid, box in boxes.items() if box.get().strip()}
        try:
            if changes:
                apply_names(folder, changes)
        except Exception as exc:
            messagebox.showerror("Could not save names", str(exc), parent=win)
            return
        outcome["value"] = "saved" if len(changes) == len(boxes) else "later"
        win.destroy()

    def dismiss():
        outcome["value"] = "dismissed"
        win.destroy()

    ttk.Button(buttons, text="Don't ask for this meeting", command=dismiss).grid(row=0, column=0, padx=4)
    ttk.Button(buttons, text="Later", command=win.destroy).grid(row=0, column=1, padx=4)
    ttk.Button(buttons, text="Save", command=save).grid(row=0, column=2, padx=4)
    win.bind("<Return>", lambda _e: save())
    win.mainloop()

    mark_prompted(folder, outcome["value"])
    return outcome["value"]
