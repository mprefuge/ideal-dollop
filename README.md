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

Phase 1 (this): manual record, local GPU transcription, per-meeting folders. Phase 2: detect meetings automatically from which app opens the mic, tray icon, and read-only calendar matching for titles and attendees. Phase 3: speaker diarization within the Others channel, summaries and action items. See `docs/architecture.md`.

## Tests

```
pip install -e .[dev]
pytest
```

The tests cover chunking, resampling, clock padding, echo removal, merging, storage and purge rules with a fake model; they need no sound card or GPU.
