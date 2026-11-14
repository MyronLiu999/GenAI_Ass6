"""Push channels: a Windows toast on this machine, and optionally ntfy.sh for the phone (subscribe to your own
hard-to-guess topic in the ntfy app). Third-party text never reaches a shell command line: the toast script
reads title and body from environment variables."""

from __future__ import annotations

import os
import subprocess
import sys

import requests

from equitydesk.config import NotifyCfg

TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $xml.GetElementsByTagName('text')
$t.Item(0).AppendChild($xml.CreateTextNode($env:ED_TITLE)) > $null
$t.Item(1).AppendChild($xml.CreateTextNode($env:ED_BODY)) > $null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""


def desktop(title: str, body: str) -> bool:
    if sys.platform != "win32":
        return False
    env = {**os.environ, "ED_TITLE": title[:120], "ED_BODY": body[:240]}
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", TOAST_PS], env=env,
                       capture_output=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode == 0


def ntfy(topic: str, title: str, body: str, url: str = "", priority: int = 3) -> bool:
    headers = {"Title": title.encode("utf-8"), "Priority": str(priority), "Tags": "chart_with_upwards_trend"}
    if url:
        headers["Click"] = url
    r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers=headers, timeout=15)
    return r.ok


def message(item: dict, verification: str) -> tuple[str, str]:
    tick = " ".join(item.get("tickers") or [])
    title = f"[{item['score']}] {tick + ' · ' if tick else ''}{item.get('title_t') or item['title']}"
    body = f"{item['source']} · {verification} · {item['published'][11:16]} UTC"
    if item.get("why"):
        body += f"\n{item['why']}"
    return title, body


def push(cfg: NotifyCfg, item: dict, verification: str) -> list[str]:
    """Send on every configured channel; returns the channels that accepted it."""
    title, body = message(item, verification)
    sent = []
    if cfg.desktop:
        try:
            if desktop(title, body):
                sent.append("desktop")
        except Exception:
            pass
    if cfg.ntfy_topic:
        try:
            if ntfy(cfg.ntfy_topic, title, body, item.get("url", ""), priority=5 if item["score"] >= 9 else 4):
                sent.append("ntfy")
        except Exception:
            pass
    return sent
