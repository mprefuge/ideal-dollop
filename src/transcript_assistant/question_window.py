"""The question window: the current question, every question asked so far, and the live transcript.

It runs in its own process and only reads what the live transcriber writes, plus
writes your own marks to live/question_marks.json, so closing it or ignoring it
never affects the recording. Reopen it with `questions` while the call is on.
"""

from __future__ import annotations

import json
from pathlib import Path

from .live_transcribe import read_marks, write_mark, write_questions_summary
from .questions import ANSWERED, DISMISSED, FOLLOW_UP, OPEN, effective_status, load_questions, pick_current
from .store import read_json

POLL_MS = 1000
ICONS = {OPEN: "?", ANSWERED: "✓", FOLLOW_UP: "↻", DISMISSED: "–"}
LABELS = {OPEN: "Not answered", ANSWERED: "Answered", FOLLOW_UP: "Follow-up", DISMISSED: "Not a question"}


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def tail_lines(path: Path, n: int = 4) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            raw = f.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    out = []
    for line in raw[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def run_question_window(folder: Path) -> None:
    import tkinter as tk
    from tkinter import ttk

    live = folder / "live"
    meta_path = folder / "meta.json"
    title = read_json(meta_path).get("title", folder.name) if meta_path.exists() else folder.name

    win = tk.Tk()
    win.title(f"Questions: {title}")
    win.geometry("460x560")
    win.minsize(360, 380)
    on_top = tk.BooleanVar(value=True)
    win.attributes("-topmost", True)

    body = ttk.Frame(win, padding=10)
    body.pack(fill="both", expand=True)

    ttk.Label(body, text="CURRENT QUESTION", foreground="#666").pack(anchor="w")
    current_text = ttk.Label(body, text="", font=("Segoe UI", 12, "bold"), wraplength=430, justify="left")
    current_text.pack(anchor="w", fill="x", pady=(2, 0))
    current_who = ttk.Label(body, text="", foreground="#666")
    current_who.pack(anchor="w")
    cur_buttons = ttk.Frame(body)
    cur_buttons.pack(anchor="w", pady=(4, 8))

    counts = ttk.Label(body, text="", foreground="#444")
    counts.pack(anchor="w")
    tree = ttk.Treeview(body, columns=("st", "t", "who", "q"), show="headings", height=10, selectmode="browse")
    for col, head, width, stretch in (("st", "", 24, False), ("t", "Time", 50, False),
                                      ("who", "Asked by", 80, False), ("q", "Question", 280, True)):
        tree.heading(col, text=head)
        tree.column(col, width=width, stretch=stretch, anchor="w")
    tree.tag_configure(OPEN, foreground="#b00020")
    tree.tag_configure(FOLLOW_UP, foreground="#a15c00")
    tree.tag_configure(ANSWERED, foreground="#1b7a2f")
    tree.tag_configure(DISMISSED, foreground="#999")
    tree.tag_configure("current", background="#fff4c2")
    tree.pack(fill="both", expand=True, pady=(2, 4))

    detail = ttk.Label(body, text="Select a question to see the reply that counted. Right-click to correct it.",
                       foreground="#555", wraplength=430, justify="left")
    detail.pack(anchor="w", fill="x")

    ttk.Separator(body).pack(fill="x", pady=6)
    transcript = ttk.Label(body, text="", foreground="#444", wraplength=430, justify="left")
    transcript.pack(anchor="w", fill="x")
    footer = ttk.Frame(body)
    footer.pack(fill="x", pady=(6, 0))
    status = ttk.Label(footer, text="", foreground="#888")
    status.pack(side="left")
    ttk.Checkbutton(footer, text="Keep on top", variable=on_top,
                    command=lambda: win.attributes("-topmost", on_top.get())).pack(side="right")

    view: dict = {"questions": [], "names": {}, "mtime": None, "current": None}

    def who(s: str) -> str:
        return view["names"].get(s, s)

    def mark(qid: str, st: str | None) -> None:
        write_mark(folder, qid, st)
        if view.get("ended"):
            write_questions_summary(folder)  # the transcriber already wrote it; keep it in step
        render()

    def show_detail(_e=None) -> None:
        sel = tree.selection()
        q = next((q for q in view["questions"] if sel and q.qid == sel[0]), None)
        if q is None:
            return
        st = effective_status(q, read_marks(folder))
        text = f"{q.text}\n{LABELS[st]}."
        if q.answered_by and st in (ANSWERED, FOLLOW_UP):
            text += f" {who(q.answered_by)} at {fmt_ts(q.answered_t or 0)}: “{q.answer}”"
        if q.qid in read_marks(folder):
            text += " (your mark)"
        detail.configure(text=text)

    menu = tk.Menu(win, tearoff=0)
    menu_target = {"qid": None}
    for st in (ANSWERED, OPEN, FOLLOW_UP, DISMISSED):
        menu.add_command(label=f"Mark {LABELS[st].lower()}", command=lambda st=st: mark(menu_target["qid"], st))
    menu.add_separator()
    menu.add_command(label="Use the automatic guess", command=lambda: mark(menu_target["qid"], None))

    def popup(e) -> None:
        row = tree.identify_row(e.y)
        if row:
            tree.selection_set(row)
            menu_target["qid"] = row
            menu.tk_popup(e.x_root, e.y_root)

    tree.bind("<<TreeviewSelect>>", show_detail)
    tree.bind("<Button-3>", popup)

    def render() -> None:
        marks = read_marks(folder)
        qs = view["questions"]
        now_t = view.get("now_t", 0.0)
        cur = pick_current(qs, marks, now_t)
        for w in cur_buttons.winfo_children():
            w.destroy()
        if cur is None:
            current_text.configure(text="No open question right now.")
            current_who.configure(text="")
        else:
            current_text.configure(text=cur.text)
            current_who.configure(text=f"{who(cur.asker)} at {fmt_ts(cur.t)}")
            ttk.Button(cur_buttons, text="Answered", command=lambda: mark(cur.qid, ANSWERED)).pack(side="left")
            ttk.Button(cur_buttons, text="Follow-up", command=lambda: mark(cur.qid, FOLLOW_UP)).pack(side="left",
                                                                                                padx=4)
            ttk.Button(cur_buttons, text="Not a question",
                       command=lambda: mark(cur.qid, DISMISSED)).pack(side="left")
        tally = {OPEN: 0, ANSWERED: 0, FOLLOW_UP: 0}
        selected = tree.selection()
        tree.delete(*tree.get_children())
        # Open ones first, newest at the top of each group, so what you still owe is always visible.
        order = {OPEN: 0, FOLLOW_UP: 1, ANSWERED: 2, DISMISSED: 3}
        for q in sorted(qs, key=lambda q: (order[effective_status(q, marks)], -q.t)):
            st = effective_status(q, marks)
            if st in tally:
                tally[st] += 1
            tags = (st, "current") if cur is not None and q.qid == cur.qid else (st,)
            tree.insert("", "end", iid=q.qid, values=(ICONS[st], fmt_ts(q.t), who(q.asker), q.text), tags=tags)
        if selected and tree.exists(selected[0]):
            tree.selection_set(selected[0])
        counts.configure(text=f"{sum(tally.values())} asked · {tally[OPEN]} not answered · "
                              f"{tally[FOLLOW_UP]} follow-up · {tally[ANSWERED]} answered")
        lines = tail_lines(live / "transcript.jsonl")
        transcript.configure(text="\n".join(f"{who(l['speaker'])}: {l['text']}" for l in lines)
                             or "Waiting for speech...")

    def poll() -> None:
        path = live / "questions.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime is not None and mtime != view["mtime"]:
            try:
                state = read_json(path)
                view.update(questions=load_questions(state), names=state.get("names", {}),
                            now_t=state.get("now_t", 0.0), mtime=mtime, status=state.get("status", ""))
                render()
            except (OSError, ValueError):
                pass
        try:
            ended = meta_path.exists() and read_json(meta_path).get("status") != "recording"
        except (OSError, ValueError):
            ended = view.get("ended", False)
        view["ended"] = ended
        line = view.get("status", "")
        if ended:
            line = "Call ended; questions saved to questions.md"
        status.configure(text=line)
        win.after(POLL_MS, poll)

    poll()
    win.mainloop()
