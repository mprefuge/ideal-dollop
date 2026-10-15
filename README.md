# transcript-assistant

Records any meeting that plays through this computer (Teams, Zoom, Meet, Webex, Slack, a phone call through the laptop) and transcribes it locally on an NVIDIA GPU. Nothing leaves the machine and nothing touches any online account.

It records two channels: the system output via Windows loopback, labeled **Others**, and your microphone, labeled **Me**. Each channel is transcribed separately with faster-whisper, then merged by time. When you are not on a headset, your mic also hears the speakers; mic segments that overlap an Others segment and say the same thing are dropped as echo, and the count is recorded in `meta.json`.

## Install (Windows, NVIDIA GPU)

Requires Python 3.10+ and a current NVIDIA driver. faster-whisper needs the CUDA 12 cuBLAS and cuDNN 9 libraries; the simplest route is the pip wheels:

```
py -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

The first transcription downloads the `large-v3-turbo` model (about 1.6 GB) from Hugging Face and caches it.

Check which devices will be recorded:

```
transcript-assistant devices
```

## Use

```
transcript-assistant record --title "Weekly sync"     # Ctrl+C to stop; transcribes when it stops
transcript-assistant transcribe --pending             # anything recorded but not transcribed
transcript-assistant list                             # all meetings, with capture warnings
transcript-assistant purge-audio --older-than 7       # dry run; add --yes to delete
```

### Automatic recording

```
transcript-assistant init-config      # writes config.json next to your transcripts
transcript-assistant watch            # tray icon; records meetings on its own
```

`watch` reads which apps currently hold the microphone, the same information Windows uses for its mic-in-use indicator (a read-only registry lookup). When Zoom, Teams, Webex, Slack, GoTo, RingCentral, Discord or a browser holds the mic for 5 seconds, it starts recording; when no meeting app has held it for 60 seconds, it stops and transcribes in the background. Mute doesn't count as releasing the mic in these apps, so muted stretches keep recording. The tray icon turns red while recording and has Stop recording this meeting (it won't restart until that meeting ends), Pause auto-record for 1 hour, Resume, Open transcripts folder, and Quit. A recording that a crash or shutdown cut off is picked up and transcribed the next time `watch` starts.

To start it at login, put a shortcut in `shell:startup` whose target is `<repo>\.venv\Scripts\pythonw.exe -m transcript_assistant.cli watch --consent-confirmed` (the windowless `pythonw` cannot show the consent prompt, so the flag records that you already agreed). Logs go to `watch.log` in the transcripts folder.

### Calendar matching (optional, read-only)

Add your calendars' secret ICS links to `ics_urls` in `config.json`. In Outlook on the web: Settings > Calendar > Shared calendars > Publish a calendar, choose "Can view all details", and copy the ICS link. In Google Calendar: Settings for the calendar > "Secret address in iCal format". When a recording starts, the event in progress (or starting within 10 minutes) names the folder and transcript, and its organizer and attendees go into `meta.json`; when two events overlap, the one whose join link matches the app in use wins. The links are refreshed at most every 15 minutes and cached under `.cache` with hashed file names, so a dropped connection doesn't stop matching. Treat these links like passwords: anyone with one can read that calendar.

Put `[no record]` or `#norecord` in an event's title or description and `watch` will not record that meeting (`skip_keywords` in `config.json`).

### Who said what

The mic channel is always you. Set `my_name` in `config.json` (for example `"Micah"`) to show your name instead of "Me", and list your addresses in `my_emails` so calendar matching can tell you apart from the other attendees.

By default everyone on the far end is labeled "Others". To split them into Speaker 1, Speaker 2, and so on, turn on speaker separation. It runs pyannote's diarization model locally on the GPU:

```
pip install -e .[diarize]
setx HF_TOKEN hf_...        # a free Hugging Face read token
```

Accept the terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on huggingface.co (the models are gated), then set `"diarize": true` in `config.json` or pass `--diarize`. If separation fails, the transcript is still written with the far end as one speaker and a warning.

Speakers are named automatically only when the calendar leaves no doubt: exactly one other person invited and one far-end voice heard. Everything else stays numbered, because a confident wrong name is worse than a number. Name them yourself afterwards:

```
transcript-assistant rename "<meeting folder>" "Speaker 1=Dana Lee" "Speaker 2=Sam Ortiz"
transcript-assistant rename "<meeting folder>" "Speaker 2="      # back to "Speaker 2"
```

**During the call.** With speaker separation on, a small "Who is it?" window pops up within about 10 to 20 seconds of a new voice on the far end. It has a Play button for that voice and a name box offering invitees, names already used on this call, and people from earlier meetings. Picking a name already used on the call tells it that's the same person. The window runs in its own process, so ignoring it never affects the recording. It works by comparing a voice fingerprint of each 3-second stretch of far-end audio with the voices heard so far, and announces a voice only after three consistent stretches. It can still occasionally ask twice about the same person, especially when people talk over each other; giving both the same name fixes that. When the meeting is transcribed, each final speaker is matched to those fingerprints, so the names carry into the transcript. Set `"live_naming": false` to turn the mid-call window off, or adjust `voice_match_threshold` (default 0.55) if it asks too often (lower it) or merges different people (raise it).

**After the call.** When a meeting with separated speakers finishes transcribing, a "Who is speaking?" window pops up for any speaker without a name. It shows how long each one talked and a few of their lines, and has a Play button that plays a few seconds of their voice. The name box offers the meeting's invitees first, then people you've named in earlier meetings, or you can type a new name. Giving two speakers the same name merges them, which fixes the common case of one person split in two. "Later" leaves them numbered, and "Don't ask for this meeting" stops the prompt. Reopen it any time with `transcript-assistant name-speakers "<meeting folder>"`, or set `"prompt_for_names": false` to turn the pop-up off. This only appears for speakers still unnamed after the mid-call window.

Names you set survive re-transcription. `transcript.json` keeps both the stable `speaker_id` and the display `speaker` for each line.

### Consent

`record` asks you to confirm that everyone has been told the meeting is recorded. California and several other states require every participant's consent; `--consent-confirmed` skips the prompt once you have already said so in the meeting.

If CUDA is not working, `--device cpu --compute-type int8` runs on the CPU, much more slowly. A failed transcription never deletes audio; rerun `transcribe` on the folder.

## Storage

Everything lives under `%USERPROFILE%\MeetingTranscripts` (override with `--root` or the `TA_HOME` environment variable), one folder per meeting:

```
2026-10-15_0930_weekly-sync/
  meta.json         title, times, per-channel levels, warnings, status
  transcript.md     readable, speaker-labeled, timestamped
  transcript.json   segments with start/end times
  audio/            mic_0000.flac, others_0000.flac, ... (5-minute chunks)
index.json          rebuilt from every meta.json; never edit by hand
```

Transcripts are sensitive. Keep BitLocker on and keep this folder out of any synced or shared location. `purge-audio` deletes audio only for meetings that were transcribed and not marked `--keep-audio`; transcripts are never deleted.

Capture problems are written to `meta.json` and shown at the top of the transcript instead of being hidden: a silent channel, a channel that captured much less than the meeting length, a device error, or mic dropouts.

## Roadmap

Phase 1: manual record, local GPU transcription, per-meeting folders. Phase 2: automatic detection from which app opens the mic, tray icon, read-only calendar matching. Phase 3 (this): speaker separation on the far end, calendar-based naming, `rename`. Summaries are deliberately not included. See `docs/architecture.md`.

## Tests

```
pip install -e .[dev]
pytest
```

The tests cover chunking, resampling, clock padding, echo removal, merging, storage and purge rules, mic-app detection, recurring calendar events, the watch state machine, speaker assignment, naming and rename, with fakes for the model, devices and clock; they need no sound card, GPU or Windows.
