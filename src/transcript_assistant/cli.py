"""Command line entry point: record, transcribe, list, purge-audio, devices."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .store import (audio_purge_candidates, default_root, new_meeting_dir, purge_audio, read_json,
                    rebuild_index, write_json_atomic)
from .transcribe import DEFAULT_MODEL

CONSENT_PROMPT = (
    "Recording laws differ by state. California and several other states require every\n"
    "participant's consent. Has everyone in this meeting been told it is being recorded? [y/N] "
)


def _transcribe(folders: list[Path], args) -> int:
    from .transcribe import load_model, transcribe_meeting

    print(f"Loading {args.model} on {args.device} ({args.compute_type})...")
    model = load_model(args.model, args.device, args.compute_type)
    failures = 0
    for folder in folders:
        print(f"Transcribing {folder.name}...")
        try:
            meta = transcribe_meeting(folder, model, args.model, args.language)
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

    from .recorder import Recording, capture_warnings

    root: Path = args.root
    started = datetime.now(timezone.utc)
    folder = new_meeting_dir(root, args.title, started)
    meta = {
        "title": args.title,
        "started_at": started.isoformat(),
        "status": "recording",
        "consent_confirmed": True,
        "keep_audio": args.keep_audio,
        "tool_version": __version__,
    }
    write_json_atomic(folder / "meta.json", meta)

    rec = Recording(folder / "audio")
    rec.start()
    print(f"Recording to {folder}\nPress Ctrl+C to stop.")
    limit = args.max_minutes * 60 if args.max_minutes else None
    try:
        while True:
            time.sleep(1)
            if rec.any_failed():
                print("A capture channel failed; stopping.", file=sys.stderr)
                break
            if limit and time.monotonic() - rec.started_monotonic >= limit:
                print("Reached --max-minutes; stopping.")
                break
    except KeyboardInterrupt:
        pass

    duration, stats = rec.stop()
    warnings = capture_warnings(duration, stats)
    meta.update(
        status="recorded",
        ended_at=datetime.now(timezone.utc).isoformat(),
        duration_s=round(duration, 1),
        channels=[asdict(s) for s in stats],
        capture_warnings=warnings,
        warnings=warnings,
    )
    write_json_atomic(folder / "meta.json", meta)
    for w in warnings:
        print(f"warning: {w}")

    code = 0
    if not args.no_transcribe:
        code = _transcribe([folder], args)
    rebuild_index(root)
    return code


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
