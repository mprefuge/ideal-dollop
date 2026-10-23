# transcript-assistant

Records any meeting that plays through this computer (Teams, Zoom, Meet, Webex, Slack, a phone call through the laptop) and transcribes it locally on an NVIDIA GPU. Nothing leaves the machine and nothing touches any online account.

It records two channels: the system output via Windows loopback, labeled **Others**, and your microphone, labeled **Me**. Each channel is transcribed separately with faster-whisper, then merged by time. When you are not on a headset, your mic also hears the speakers; mic segments that overlap an Others segment and say the same thing are dropped as echo, and the count is recorded in `meta.json`.

## Install (Windows, NVIDIA GPU)

Requires Python 3.10+ and a current NVIDIA driver. faster-whisper needs the CUDA 12 cuBLAS and cuDNN 9 libraries; the simplest route is the pip wheels:

```
py -3.12 -m venv .venv
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
transcript-assistant recover                          # same, plus recordings a crash left unfinished (--list to preview)
transcript-assistant list                             # all meetings, with capture warnings
transcript-assistant purge-audio --older-than 7       # dry run; add --yes to delete
```

### Automatic recording

```
transcript-assistant init-config      # writes config.json next to your transcripts
transcript-assistant watch            # tray icon; records meetings on its own
```

`watch` reads which apps currently hold the microphone, the same information Windows uses for its mic-in-use indicator (a read-only registry lookup). When Zoom, Teams, Webex, Slack, GoTo, RingCentral, Discord or a browser holds the mic for 5 seconds, it starts recording; when no meeting app has held it for 60 seconds, it stops and transcribes in the background. Mute doesn't count as releasing the mic in these apps, so muted stretches keep recording. The tray icon turns red while recording and has Stop recording this meeting (it won't restart until that meeting ends), Pause auto-record for 1 hour, Resume, Show questions window, Transcribe missed meetings, Open transcripts folder, and Quit. Transcribe missed meetings (or `recover` from the command line) finds every meeting that was recorded but never transcribed, including ones a crash left marked as still recording, and transcribes them; a recording whose audio changed in the last 2 minutes is left alone in case it is still going. A recording that a crash or shutdown cut off is picked up and transcribed the next time `watch` starts.

To start it at login, put a shortcut in `shell:startup` whose target is `<repo>\.venv\Scripts\pythonw.exe -m transcript_assistant.cli watch --consent-confirmed` (the windowless `pythonw` cannot show the consent prompt, so the flag records that you already agreed). Logs go to `watch.log` in the transcripts folder.

### Calendar matching (optional, read-only)

Add your calendars' secret ICS links to `ics_urls` in `config.json`. In Outlook on the web: Settings > Calendar > Shared calendars > Publish a calendar, choose "Can view all details", and copy the ICS link. In Google Calendar: Settings for the calendar > "Secret address in iCal format". When a recording starts, the event in progress (or starting within 10 minutes) names the folder and transcript, and its organizer and attendees go into `meta.json`; when two events overlap, the one whose join link matches the app in use wins. The links are refreshed at most every 15 minutes and cached under `.cache` with hashed file names, so a dropped connection doesn't stop matching. Treat these links like passwords: anyone with one can read that calendar.

Put `[no record]` or `#norecord` in an event's title or description and `watch` will not record that meeting (`skip_keywords` in `config.json`).

### Who said what

The mic channel is always you. Set `my_name` in `config.json` (for example `"Micah"`) to show your name instead of "Me", and list your addresses in `my_emails` so calendar matching can tell you apart from the other attendees.

By default everyone on the far end is labeled "Others". To split them into Speaker 1, Speaker 2, and so on, turn on speaker separation. It runs pyannote's diarization model locally on the GPU:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build; plain pip gets CPU-only torch on Windows
pip install -e .[diarize]
setx HF_TOKEN hf_...        # a free Hugging Face read token; open a new terminal afterwards
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

### Live transcript and question tracker

While a meeting records (by hand or through `watch`), both channels are also transcribed live and a small "Questions" window opens. It shows the current question at the top, every question asked so far below it (not answered first, then follow-up, then answered), and the last few lines of the live transcript at the bottom. Closing it never affects the recording; reopen it with `transcript-assistant questions`, or from the tray menu's Show questions window.

Everything runs on this computer. The live text comes from a smaller Whisper model (`live_model`, default `small`, about 500 MB of GPU memory on top of the main model) so it keeps up in real time; the saved transcript is still made by the large model after the call. Questions are found and tracked with plain rules, not an AI model:

- A sentence is a question when Whisper ends it with "?" or it opens like one ("can you", "do we", "is there", "what", ...). Tags like "right?" or "you know?" are ignored.
- The current question is the newest one still open. Once it's answered, an earlier open question from the last 10 minutes becomes current again.
- A question counts as answered when someone other than the asker replies with a few words that aren't themselves a question, or with a yes/no. A reply like "I'll get back to you" or "not sure" marks it follow-up instead.
- With speaker separation off, everyone on the far end shares the "Others" label, so a reply from "Others" to an "Others" question counts only after a pause and with more words.

These are guesses. Use the buttons under the current question, or right-click any row, to mark it answered, not answered, follow-up, or not a question; your marks always win and are kept. When the call ends, `questions.md` in the meeting folder lists what was asked, grouped by status, with the reply that counted.

Set `"live_transcribe": false` in `config.json` to turn all of this off, `"question_window": false` to keep the live transcript (in `live/transcript.jsonl`) without the window, or `"live_model": "large-v3-turbo"` for better live accuracy if the GPU has 6 GB or more.

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
