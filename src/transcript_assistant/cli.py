"""Command line entry point: record, watch, transcribe, rename, name-speakers, list, purge-audio, devices."""

from __future__ import annotations

import argparse
import sys
import time
import logging
import threading
from pathlib import Path

from .config import load_config, write_default_config
from .store import audio_purge_candidates, default_root, purge_audio, read_json, rebuild_index
from .transcribe import DEFAULT_MODEL

CONSENT_PROMPT = (
    "Recording laws differ by state. California and several other states require every\n"
    "participant's consent. Has everyone in this meeting been told it is being recorded? [y/N] "
)


def _load_engines(args, cfg):
    """Whisper model plus, when enabled, a diarizer callable. Loaded once and reused."""
    from .transcribe import load_model

    print(f"Loading {args.model} on {args.device} ({args.compute_type})...")
    model = load_model(args.model, args.device, args.compute_type)
    diarizer = embedder = None
    if args.diarize or cfg.diarize:
        from .diarize import load_pipeline, run_pipeline
        from .live import load_embedder

        print("Loading speaker separation model...")
        pipeline = load_pipeline(device=args.device)
        diarizer = lambda audio: run_pipeline(pipeline, audio)  # noqa: E731
        try:
            embedder = load_embedder(device=args.device)  # matches names given during the call
        except Exception as exc:
            print(f"warning: voice fingerprint model unavailable ({exc}); names from the call won't carry over",
                  file=sys.stderr)
    return model, diarizer, embedder


def _transcribe_one(folder: Path, engines, args, cfg) -> dict:
    from .transcribe import transcribe_meeting

    model, diarizer, embedder = engines
    return transcribe_meeting(folder, model, args.model, args.language or cfg.language,
                              diarizer=diarizer, my_name=cfg.my_name, my_emails=cfg.my_emails,
                              embedder=embedder, voice_threshold=cfg.voice_match_threshold)


def _live_factory(args, cfg, root: Path):
    """Factory for mid-call naming, or None when it's off. The fingerprint model loads on first use."""
    if not ((args.diarize or cfg.diarize) and cfg.live_naming):
        return None
    from .live import LiveSpeakers, launch_voice_dialog, load_embedder

    holder: dict = {}
    lock = threading.Lock()

    def embed(audio):
        with lock:
            if "fn" not in holder:
                holder["fn"] = load_embedder(device=args.device)
        return holder["fn"](audio)

    return lambda folder: LiveSpeakers(folder, embed, on_new_voice=lambda f, vid: launch_voice_dialog(root, f, vid),
                                       threshold=cfg.voice_match_threshold)


def _transcribe(folders: list[Path], args) -> int:
    cfg = load_config(args.root)
    engines = _load_engines(args, cfg)
    failures = 0
    for folder in folders:
        print(f"Transcribing {folder.name}...")
        try:
            meta = _transcribe_one(folder, engines, args, cfg)
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}. Audio is untouched; retry with "
                  f"`transcript-assistant transcribe \"{folder}\"`.", file=sys.stderr)
            continue
        print(f"  {folder / 'transcript.md'}")
        for w in meta.get("warnings", []):
            print(f"  warning: {w}")
    return 1 if failures else 0


def cmd_record(args) -> int:
    if not args.consent_confirmed:
        if input(CONSENT_PROMPT).strip().lower() not in ("y", "yes"):
            print("Not recording.")
            return 1

    from .session import finish_session, start_session

    root: Path = args.root
    session = start_session(root, args.title, {"consent_confirmed": True, "keep_audio": args.keep_audio,
                                               "trigger": "manual"},
                            live_factory=_live_factory(args, load_config(root), root))
    rec = session.recording
    print(f"Recording to {session.folder}\nPress Ctrl+C to stop.")
    limit = args.max_minutes * 60 if args.max_minutes else None
    reason = "stopped by user"
    try:
        while True:
            time.sleep(1)
            if rec.any_failed():
                print("A capture channel failed; stopping.", file=sys.stderr)
                reason = "capture channel failed"
                break
            if limit and time.monotonic() - rec.started_monotonic >= limit:
                print("Reached --max-minutes; stopping.")
                reason = "reached --max-minutes"
                break
    except KeyboardInterrupt:
        pass

    meta = finish_session(session, reason)
    for w in meta["warnings"]:
        print(f"warning: {w}")
    folder = session.folder

    code = 0
    if not args.no_transcribe:
        code = _transcribe([folder], args)
        cfg = load_config(root)
        if code == 0 and cfg.prompt_for_names:
            from .speakers import should_prompt

            if should_prompt(folder):
                from .speaker_dialog import run_dialog

                run_dialog(root, folder, cfg.my_emails)
    rebuild_index(root)
    return code


WATCH_CONSENT_PROMPT = (
    "Auto-recording will capture every meeting that opens your microphone.\n"
    "California and several other states require every participant's consent, so you\n"
    "agree to tell participants in each meeting that it is being recorded, or to mark\n"
    "the calendar event with a skip keyword (see config.json). Agree? [y/N] "
)


def cmd_watch(args) -> int:
    from .calendar_match import CalendarSource
    from .detector import default_mic_reader
    from .speakers import launch_dialog
    from .watcher import TranscriptionWorker, Watcher, pending_meetings, recover_interrupted

    if not args.consent_confirmed:
        if input(WATCH_CONSENT_PROMPT).strip().lower() not in ("y", "yes"):
            print("Not watching.")
            return 1

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(root / "watch.log", encoding="utf-8"), logging.StreamHandler()],
    )
    cfg = load_config(root)
    calendar = (CalendarSource(cfg.ics_urls, root / ".cache", cfg.calendar_refresh_minutes)
                if cfg.ics_urls else None)

    watcher = Watcher(root, cfg, default_mic_reader(), calendar, live_factory=_live_factory(args, cfg, root))
    worker = TranscriptionWorker(
        root,
        load_model=lambda: _load_engines(args, cfg),
        transcribe=lambda folder, engines: _transcribe_one(folder, engines, args, cfg),
        notify=lambda msg: watcher.notify(msg),
        after=(lambda folder: launch_dialog(root, folder)) if cfg.prompt_for_names else None,
    )
    watcher.on_recorded = worker.submit
    worker.start()

    for folder in recover_interrupted(root):
        logging.warning("Recovered interrupted recording %s", folder.name)
    for folder in pending_meetings(root):
        worker.submit(folder)
    rebuild_index(root)

    stop_event = threading.Event()
    if args.no_tray:
        print("Watching for meetings. Press Ctrl+C to quit.")
        try:
            watcher.run(stop_event)
        except KeyboardInterrupt:
            watcher.shutdown()
    else:
        from .tray import run_tray

        run_tray(watcher, root, stop_event)
    if not worker.queue.empty():
        print("Some meetings are still waiting for transcription; "
              "they will be picked up next time, or run `transcript-assistant transcribe --pending`.")
    return 0


def cmd_init_config(args) -> int:
    path, created = write_default_config(args.root)
    print(("Created " if created else "Already exists, left unchanged: ") + str(path))
    return 0


def cmd_rename(args) -> int:
    from .speakers import apply_names, load_segments

    folder = Path(args.folder)
    if not (folder / "transcript.json").exists():
        print("This meeting has not been transcribed yet.", file=sys.stderr)
        return 1
    changes = {}
    for pair in args.names:
        sid, sep, name = pair.partition("=")
        if not sep:
            print(f"Expected ID=NAME, got {pair!r}", file=sys.stderr)
            return 1
        changes[sid.strip()] = name.strip()
    try:
        names = apply_names(folder, changes)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    for sid in sorted({s.speaker for s in load_segments(folder)}):
        print(f"{sid}: {names.get(sid, sid)}")
    return 0


def cmd_name_voice(args) -> int:
    from .live_dialog import run_voice_dialog

    cfg = load_config(args.root)
    name = run_voice_dialog(args.root, Path(args.folder), args.voice_id, cfg.my_emails)
    print(f"{args.voice_id}: {name or '(skipped)'}")
    return 0


def cmd_name_speakers(args) -> int:
    from .speaker_dialog import run_dialog

    cfg = load_config(args.root)
    outcome = run_dialog(args.root, Path(args.folder), cfg.my_emails)
    print(f"Speaker names: {outcome}")
    return 0


def cmd_transcribe(args) -> int:
    root: Path = args.root
    if args.pending:
        folders = [p.parent for p in sorted(root.glob("*/meta.json"))
                   if read_json(p).get("status") == "recorded"]
    else:
        folders = [Path(f) for f in args.folders]
    if not folders:
        print("Nothing to transcribe.")
        return 0
    code = _transcribe(folders, args)
    rebuild_index(root)
    return code


def cmd_list(args) -> int:
    for e in rebuild_index(args.root):
        flag = f"  [{len(e['warnings'])} warning(s)]" if e.get("warnings") else ""
        mins = f"{e['duration_s'] / 60:.0f} min" if e.get("duration_s") else "?"
        print(f"{e.get('started_at', '?')[:16]}  {e.get('status', '?'):<11} {mins:>7}  {e['folder']}{flag}")
    return 0


def cmd_purge_audio(args) -> int:
    dirs = audio_purge_candidates(args.root, args.older_than)
    if not dirs:
        print("No audio to purge.")
        return 0
    for d in dirs:
        print(("Deleting " if args.yes else "Would delete ") + str(d))
    if not args.yes:
        print("Dry run. Re-run with --yes to delete. Transcripts are never deleted.")
        return 0
    purge_audio(dirs)
    rebuild_index(args.root)
    return 0


def cmd_devices(_args) -> int:
    import soundcard as sc

    print(f"Default speaker (recorded as 'others' via loopback): {sc.default_speaker().name}")
    print(f"Default microphone (recorded as 'mic'): {sc.default_microphone().name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="transcript-assistant", description=__doc__)
    p.add_argument("--root", type=Path, default=default_root(),
                   help="Where meetings are stored (default: TA_HOME or ~/MeetingTranscripts)")
    sub = p.add_subparsers(dest="command", required=True)

    def model_opts(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--device", default="cuda", help="cuda (default) or cpu")
        sp.add_argument("--compute-type", default="float16", help="float16 on GPU; int8 on CPU")
        sp.add_argument("--language", default=None, help="e.g. en; auto-detected if omitted")
        sp.add_argument("--diarize", action="store_true",
                        help="Separate individual speakers on the far end (also: diarize in config.json)")

    r = sub.add_parser("record", help="Record a meeting, then transcribe it")
    r.add_argument("--title", default="Meeting")
    r.add_argument("--consent-confirmed", action="store_true", help="Skip the consent prompt")
    r.add_argument("--no-transcribe", action="store_true")
    r.add_argument("--keep-audio", action="store_true", help="Exclude this meeting's audio from purge-audio")
    r.add_argument("--max-minutes", type=int, default=240, help="Safety stop (0 for none)")
    model_opts(r)
    r.set_defaults(func=cmd_record)

    t = sub.add_parser("transcribe", help="Transcribe recorded meetings")
    t.add_argument("folders", nargs="*")
    t.add_argument("--pending", action="store_true", help="Every meeting recorded but not yet transcribed")
    model_opts(t)
    t.set_defaults(func=cmd_transcribe)

    w = sub.add_parser("watch", help="Record meetings automatically when a meeting app opens the mic")
    w.add_argument("--consent-confirmed", action="store_true", help="Skip the consent prompt")
    w.add_argument("--no-tray", action="store_true", help="Run in the console without a tray icon")
    model_opts(w)
    w.set_defaults(func=cmd_watch)

    rn = sub.add_parser("rename", help='Name speakers, e.g. rename FOLDER "Speaker 1=Dana Lee"')
    rn.add_argument("folder")
    rn.add_argument("names", nargs="+", metavar="ID=NAME")
    rn.set_defaults(func=cmd_rename)

    ns = sub.add_parser("name-speakers", help="Open the who-is-speaking dialog for a meeting")
    ns.add_argument("folder")
    ns.set_defaults(func=cmd_name_speakers)

    nv = sub.add_parser("name-voice", help=argparse.SUPPRESS)  # opened by live naming during a call
    nv.add_argument("folder")
    nv.add_argument("voice_id")
    nv.set_defaults(func=cmd_name_voice)

    ic = sub.add_parser("init-config", help="Write a default config.json if none exists")
    ic.set_defaults(func=cmd_init_config)

    ls = sub.add_parser("list", help="List meetings and rebuild index.json")
    ls.set_defaults(func=cmd_list)

    pa = sub.add_parser("purge-audio", help="Delete audio of transcribed meetings (dry run by default)")
    pa.add_argument("--older-than", type=int, default=7, metavar="DAYS")
    pa.add_argument("--yes", action="store_true")
    pa.set_defaults(func=cmd_purge_audio)

    d = sub.add_parser("devices", help="Show which audio devices will be recorded")
    d.set_defaults(func=cmd_devices)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "transcribe" and not args.pending and not args.folders:
        build_parser().error("give meeting folders or --pending")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
