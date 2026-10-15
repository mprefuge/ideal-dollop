"""Detect meetings by which apps currently hold the microphone open.

Windows records every app's microphone use in the registry for its privacy
indicator: under CapabilityAccessManager\\ConsentStore\\microphone, an app whose
LastUsedTimeStop is 0 has the microphone open right now. Reading that is
read-only, needs no admin rights, and works for any meeting app.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

CONSENT_KEY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"

# Lower-cased substrings of the app id (exe path or package name) -> platform label.
MEETING_APPS = {
    "zoom.exe": "Zoom",
    "ms-teams.exe": "Teams",
    "msteams_": "Teams",  # new Teams, packaged
    "teams.exe": "Teams",
    "webex": "Webex",
    "ciscocollabhost": "Webex",
    "atmgr.exe": "Webex",
    "slack.exe": "Slack",
    "ringcentral": "RingCentral",
    "gotomeeting": "GoTo",
    "g2mcomm.exe": "GoTo",
    "discord.exe": "Discord",
}

# Browsers rarely use the mic for anything but calls (Meet, Teams web, Zoom web).
BROWSERS = {
    "chrome.exe": "Browser",
    "msedge.exe": "Browser",
    "firefox.exe": "Browser",
    "brave.exe": "Browser",
    "opera.exe": "Browser",
}

# Our own recorder opens the mic too; counting it would keep a recording alive forever.
SELF_APPS = ("python.exe", "pythonw.exe", "transcript-assistant.exe")


@dataclass(frozen=True)
class MicUse:
    app_id: str
    in_use: bool


def classify(app_id: str, include_browsers: bool = True, extra_apps: dict[str, str] | None = None) -> str | None:
    """Platform label for a meeting app, or None if this app isn't one we record."""
    key = app_id.lower().replace("#", "\\")
    if key.endswith(SELF_APPS):
        return None
    table = dict(MEETING_APPS)
    if include_browsers:
        table.update(BROWSERS)
    if extra_apps:
        table.update({k.lower(): v for k, v in extra_apps.items()})
    for needle, label in table.items():
        if needle in key:
            return label
    return None


def active_meeting_apps(uses: Iterable[MicUse], include_browsers: bool = True,
                        extra_apps: dict[str, str] | None = None) -> list[str]:
    """Sorted, de-duplicated platform labels of meeting apps holding the mic now."""
    labels = {classify(u.app_id, include_browsers, extra_apps) for u in uses if u.in_use}
    labels.discard(None)
    return sorted(labels)


def read_windows_mic_uses() -> list[MicUse]:  # pragma: no cover - Windows only
    import winreg

    uses: list[MicUse] = []

    def scan(path: str, prefix: str = "") -> None:
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path)
        except OSError:
            return
        with key:
            i = 0
            while True:
                try:
                    name = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                if name == "NonPackaged":
                    continue
                try:
                    with winreg.OpenKey(key, name) as sub:
                        start, _ = winreg.QueryValueEx(sub, "LastUsedTimeStart")
                        stop, _ = winreg.QueryValueEx(sub, "LastUsedTimeStop")
                except OSError:
                    continue
                uses.append(MicUse(prefix + name, bool(start) and stop == 0))

    scan(CONSENT_KEY)
    scan(CONSENT_KEY + r"\NonPackaged")
    return uses


def default_mic_reader() -> Callable[[], list[MicUse]]:
    if sys.platform != "win32":
        raise RuntimeError("Automatic meeting detection currently supports Windows only.")
    return read_windows_mic_uses
