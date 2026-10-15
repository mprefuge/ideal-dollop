# Transcript Assistant: architecture proposal

Status: proposal, no code written, nothing touches any account yet. 2026-10-15.

## Recommendation in one paragraph

Capture meeting audio on Micah's own computer at the operating-system level, transcribe it locally, and name each transcript by matching its time window against the calendar. This is the only design that honors "regardless of system used": it works the same for Teams, Zoom, Meet, Webex, Slack huddles, a phone call routed through the laptop, or a platform nobody has thought of yet, and it needs no admin consent from any tenant. Native platform transcripts become an optional quality upgrade layered on later, not the backbone.

## Why not lead with native transcripts (the original default)

The coordinator's starting default was "pull native transcripts from Teams/Zoom/Meet, fall back to local capture." What I found argues for flipping that.

The personal Google account has no online meetings to pull from. In the last 30 days the primary and Shared calendars hold only church services and member meetings with no conference links, and Drive has no Meet recordings, Gemini notes or transcript files (checked read-only via the Calendar and Drive connectors). Meet transcripts also require a paid Workspace edition, which a gmail.com account is not.

The work meetings most likely live in a university Microsoft 365 tenant that this project cannot reach. Pulling Teams transcripts through Graph needs `OnlineMeetingTranscript.Read.All` or per-meeting organizer access, which university tenants usually block for third-party apps, and it only works when someone turned transcription on in that meeting. Zoom cloud transcripts need a paid plan and the host's recording. Each native source covers only the meetings where someone else made the right choice, so it can never reach "all meetings."

The `mprefuge/org-platform` repo was checked as a candidate for "my system". It is the Stripe / Salesforce / QuickBooks integration stack and has nothing meeting-related, so "my system" is read here as "my computer."

## Components

**1. Meeting detector.** A small background service watches which processes hold an active microphone or audio session (on Windows, WASAPI audio session enumeration; on macOS, Core Audio process taps). When a known meeting process (Teams, Zoom, Webex, Slack, or a browser tab on meet.google.com) opens the mic, it starts a capture; when the session goes idle for a couple of minutes it stops. A global hotkey and tray icon allow manual start, stop and "don't record this one."

**2. Recorder.** Two separate channels: system output via loopback (everyone else) and the microphone (Micah). Keeping them separate gives a free and reliable "me versus them" speaker split before any diarization model runs. Audio is written in short FLAC chunks so a crash loses seconds, not the meeting.

**3. Transcriber.** Default is local `faster-whisper` (large-v3-turbo) so audio never leaves the machine; `pyannote` diarizes the remote channel into Speaker 1, 2, and so on. If the machine has no usable GPU, a cloud speech API (Deepgram or AssemblyAI) is the fallback, at roughly $0.25 to $0.40 per meeting hour, and that would be a data-handling decision to confirm first.

**4. Calendar matcher (read-only).** Matches the recording window to a calendar event to set the title, attendees and meeting link, and maps "Speaker 2" guesses onto attendee names when there are only two or three people. Google Calendar is reachable today; Outlook would need a read-only Microsoft Graph `Calendars.Read` grant, or an ICS export, if the work calendar is the source.

**5. Store.** One folder per meeting: `transcript.md` (readable, timestamped, speaker-labeled), `transcript.json` (segments with timings), `meta.json` (event, attendees, platform, durations), and the audio, which is deleted after a retention window unless kept. A single `index.json` makes everything searchable. Decided 2026-10-15: the store is a local folder only, with no cloud sync. Backup is Micah's call.

**6. Optional afterwards.** A Claude pass that writes a summary, decisions and action items into the same folder; native transcript ingestion for meetings where Teams or Zoom produced one, used to correct names and words; push of action items to Salesforce or a task tool.

## Risks I want on the record

**Recording consent is the biggest risk, not the code.** Kentucky is a one-party-consent state, so recording a call Micah is on is generally lawful there. California is an all-party-consent state (Penal Code 632), and the work calendar points to a California university, so any meeting with a participant in California needs every party's consent. University policy may add its own rules. The tool should therefore default to announcing itself: a recording indicator, a one-click "post disclosure to chat" for the platform in use, and a per-meeting skip. This is not legal advice, and it is worth a check with the university before turning on auto-start for work meetings.

**Sensitive content at rest.** Transcripts will hold donor, student and personnel conversations. The store should be encrypted at rest (BitLocker or FileVault at minimum), excluded from any shared or public sync, and audio should be purged after transcription unless deliberately kept.

**Silent failure.** A recorder that quietly stops capturing is worse than none. Each meeting's `meta.json` records captured duration versus calendar duration and per-channel audio levels, and a mismatch is flagged in the index rather than hidden.

## What touches accounts

Nothing writes to any account in the core design. Calendar access is read-only. All writes are local files. Anything that would write elsewhere (posting a disclosure into a meeting chat, pushing action items to Salesforce, uploading to cloud speech-to-text) is opt-in and will be flagged before it is built.

## Build order

Phase 1 is the recorder, local transcription and the per-meeting folder, run by hand from the tray. Phase 2 adds auto-detection and calendar matching. Phase 3 adds summaries and native-transcript enrichment. Phase 1 alone is useful and is the first PR, in a new repo (proposed name `transcript-assistant`), Python, packaged to run at login.

## Open decisions for Micah

1. ~~Which computer and GPU~~ Decided: Windows with an NVIDIA GPU, transcription runs locally. Own repo, not org-platform.
2. ~~Where transcripts should live~~ Decided: local only.
