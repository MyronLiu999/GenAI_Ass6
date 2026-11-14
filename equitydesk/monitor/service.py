"""The monitor loop. One tick: read the control state, collect every source in parallel, store what is new,
cluster duplicate stories, score new stories, push the important ones, write at most one event brief, refresh
quotes / sentiment / odds and the overview when they are due, and record a heartbeat.

Control lives in the store (kv 'control': running | paused | stopped), so the UI can pause or stop the service
from another process. Collection never stops for money; every model call stops once the day's spend reaches
`monitor.daily_cost_ceiling_usd`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from equitydesk.config import ROOT, Settings, load_env, load_settings
from equitydesk.harness.llm import LLM, BudgetExceeded, ConfigError, ModelError
from equitydesk.harness.trace import Tracer
from equitydesk.monitor import notify, reports, signals, sources
from equitydesk.monitor.store import MonitorStore, utcnow
from equitydesk.monitor.triage import (BATCH, cluster, match_tickers, overview, score_batch, translated_duplicates,
                                       verification)

SCORE_PER_TICK = 100      # bounds the work (and the spend) of one tick after a long pause


def _age_min(at: str | None) -> float:
    if not at:
        return float("inf")
    return (datetime.now(timezone.utc) - datetime.fromisoformat(at.replace("Z", "+00:00"))).total_seconds() / 60


class MonitorService:
    def __init__(self, settings: Settings, store: MonitorStore | None = None, *, backend_factory=None,
                 fetch: Callable = sources.fetch, fetch_yahoo: Callable = sources.fetch_yahoo,
                 quotes: Callable = signals.quotes, reddit: Callable = signals.reddit_mentions,
                 stocktwits: Callable = signals.stocktwits, odds: Callable = signals.predictions,
                 push: Callable = notify.push):
        self.s, self.cfg = settings, settings.monitor
        self.store = store or MonitorStore(settings.monitor_path)
        self.factory = backend_factory
        self.fetch, self.fetch_yahoo = fetch, fetch_yahoo
        self.quotes_fn, self.reddit_fn, self.st_fn, self.odds_fn, self.push_fn = quotes, reddit, stocktwits, odds, push

    # -- control --------------------------------------------------------------------------------
    def state(self) -> str:
        return (self.store.get("control") or {}).get("state", "running")

    def set_state(self, state: str) -> None:
        self.store.put("control", {"state": state})

    def beat(self, state: str, **extra) -> None:
        self.store.put("heartbeat", {"pid": os.getpid(), "state": state, "interval_s": self.cfg.interval_s, **extra})

    # -- one tick -------------------------------------------------------------------------------
    def tick(self) -> dict:
        state = self.state()
        self.beat(state)
        if state != "running":
            return {"state": state}
        watch = self.store.watch()
        summary = {"state": "running", **self.collect(watch)}
        summary.update(self.model_steps(watch))
        summary["signals"] = self.refresh_signals(watch)
        if _age_min((self.store.get_with_time("pruned")[1])) > 24 * 60:
            self.store.prune(self.cfg.keep_days)
            self.store.put("pruned", True)
        self.store.put("last_collect", {k: v for k, v in summary.items() if k != "signals"})
        self.beat("running")
        return summary

    def collect(self, watch: list[dict]) -> dict:
        jobs = {s.key: (lambda s=s: self.fetch(s)) for s in sources.enabled(self.cfg.sources)}
        for w in watch:
            jobs[f"yahoo_{w['ticker']}"] = lambda t=w["ticker"]: self.fetch_yahoo(t)
        health = self.store.get("health") or {}
        items, errors = [], {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {k: pool.submit(fn) for k, fn in jobs.items()}
            for k, f in futs.items():
                h = health.get(k, {})
                try:
                    got = f.result()
                    items += got
                    health[k] = {**h, "ok_at": utcnow(), "n": len(got), "error": None}
                except Exception as e:
                    errors[k] = f"{type(e).__name__}: {e}"[:200]
                    health[k] = {**h, "error": errors[k], "error_at": utcnow()}
        self.store.put("health", health)
        heads = self.store.recent_heads(36)
        new = self.store.add_items(items)
        members = 0
        for it, head in cluster(new, heads):
            if head:
                self.store.set_cluster(it["key"], head)
                members += 1
        return {"fetched": len(items), "new": len(new), "duplicates": members, "errors": errors}

    def notify_cfg(self):
        """settings.yaml's notify block, with what the user changed on the Monitor page on top."""
        return self.cfg.notify.model_copy(update=self.store.get("notify") or {})

    def engine(self) -> str:
        return self.store.get("engine") or self.cfg.engine

    def model_settings(self) -> Settings:
        """With the claude_code engine, the monitor's three model roles run on the Claude Code CLI."""
        if self.engine() != "claude_code":
            return self.s
        m = self.cfg.claude_models
        roles = {**self.s.roles,
                 "monitor": {"provider": "claude_code", "name": m.get("score", "haiku"), "timeout_s": 120},
                 "monitor_overview": {"provider": "claude_code", "name": m.get("overview", "sonnet"), "timeout_s": 180},
                 "monitor_report": {"provider": "claude_code", "name": m.get("report", "sonnet"), "timeout_s": 180}}
        return self.s.model_copy(update={"roles": roles})

    def _llm(self) -> LLM | None:
        left = self.cfg.daily_cost_ceiling_usd - self.store.spend_today()
        if left <= 0 and self.engine() == "openai":
            return None
        tracer = Tracer(f"MON-{time.strftime('%Y%m%d')}", self.s.runs_dir)
        return LLM(self.model_settings(), tracer, backend_factory=self.factory, token_ceiling=10_000_000,
                   cost_ceiling=max(left, 0.01) if self.engine() == "openai" else 1e9)

    def model_steps(self, watch: list[dict]) -> dict:
        llm = self._llm()
        if llm is None:
            self.store.put("budget_note", f"daily cost ceiling ${self.cfg.daily_cost_ceiling_usd:.2f} reached: "
                                          "collection continues, scoring and reports resume tomorrow")
            return {"scored": 0, "pushed": 0, "report": None, "overview": False, "budget_stop": True}
        self.store.put("budget_note", None)
        out = {"scored": 0, "pushed": 0, "report": None, "overview": False, "budget_stop": False}
        try:
            out["scored"] = self.score(llm, watch)
            out["pushed"] = self.push_due(watch)
            out["report"] = self.report_due(llm, watch)
            out["overview"] = self.overview_due(llm, watch)
        except BudgetExceeded:
            out["budget_stop"] = True
        except (ConfigError, ModelError) as e:
            out["model_error"] = str(e)[:300]
            self.store.put("model_error", {"error": str(e)[:300]})
        finally:
            self.store.add_spend(llm.meter.cost_usd)
        return out

    def score(self, llm: LLM, watch: list[dict]) -> int:
        todo = self.store.unscored(SCORE_PER_TICK)
        for it in todo:
            # a Yahoo per-ticker feed only hints at its ticker (it carries unrelated stories too): the ticker is
            # kept if the text names the company or the model links the story to it
            it["matched"] = match_tickers(f"{it['title']} {it.get('summary', '')}", watch)
        n, done = 0, []
        for i in range(0, len(todo), BATCH):
            for r in score_batch(llm, todo[i:i + BATCH], watch, self.cfg.language):
                self.store.set_score(r["key"], r["score"], r["category"], r["tickers"], r["title"], r["why"], r["rumor"])
                done.append(r["key"])
                n += 1
        if done:   # the same story in two languages or two wordings: only the first stays a head
            fresh = [self.store.get_item(k) for k in done]
            recent = [i for i in self.store.items(hours=24) if i.get("scored_at")]
            for key, head in translated_duplicates([f for f in fresh if f], recent):
                self.store.set_cluster(key, head)
        return n

    def push_due(self, watch: list[dict]) -> int:
        holdings = {w["ticker"] for w in watch if w["holding"]}
        log = self.store.get("pushes") or []
        hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        room = self.cfg.max_pushes_per_hour - sum(1 for p in log if p["at"] >= hour_ago and p.get("sent", True))
        n = 0
        due = sorted(self.store.to_push(self.cfg.push_min_score, self.cfg.push_holding_min_score, holdings),
                     key=lambda i: -i["score"])
        warm = self.store.get("warm_start_done") is None   # first ever tick: the backlog is not news to push
        if warm:
            self.store.put("warm_start_done", True)
        for it in due:
            if room <= 0 or warm:   # over the hourly cap (or backlog): held back, never re-sent later in a burst
                self.store.mark_pushed(it["key"], sent=False)
                log = (self.store.get("pushes") or [])[-199:]
                self.store.put("pushes", log + [{"key": it["key"], "at": utcnow(), "channels": [], "sent": False}])
                continue
            room -= 1
            sent = self.push_fn(self.notify_cfg(), it, verification(it))
            self.store.mark_pushed(it["key"])       # once, whether or not a channel accepted it
            log = (self.store.get("pushes") or [])[-199:]
            self.store.put("pushes", log + [{"key": it["key"], "at": utcnow(), "channels": sent}])
            n += 1
        return n

    def report_due(self, llm: LLM, watch: list[dict]) -> str | None:
        if self.store.reports_today() >= self.cfg.max_reports_per_day:
            return None
        cands = [i for i in self.store.items(hours=6, min_score=self.cfg.report_min_score) if not i.get("report_id")]
        if not cands:
            return None
        it = max(cands, key=lambda i: (i["score"], i["published"]))
        rep = reports.write_report(llm, it, watch, self.cfg.language, datetime.now().strftime("%Y-%m-%d"))
        self.store.add_report(rep["id"], it["key"], rep["title"], rep["body"], rep["sources"], rep["cost"])
        return rep["id"]

    def overview_due(self, llm: LLM, watch: list[dict], force: bool = False) -> bool:
        _, at = self.store.get_with_time("overview")
        items = [i for i in self.store.items(hours=24, min_score=5) if i.get("scored_at")]
        newer = [i for i in items if not at or i["scored_at"] > at]
        if not force and (_age_min(at) < self.cfg.overview_every_min or not newer):
            return False
        if not items:
            return False
        items = sorted(items, key=lambda i: (-i["score"], i["published"]))[:60]
        ov = overview(llm, items, self.store.get("quotes") or {}, self.store.get("predictions") or [], watch,
                      self.cfg.language)
        self.store.put("overview", ov)
        return True

    def refresh_signals(self, watch: list[dict]) -> dict:
        tickers = [w["ticker"] for w in watch]
        done = {}
        plan = (("quotes", self.cfg.quotes_every_min, lambda: self.quotes_fn(tickers)),
                ("reddit", self.cfg.social_every_min, lambda: self.reddit_fn()),
                ("stocktwits", self.cfg.social_every_min,
                 lambda: {t: self.st_fn(t) for t in tickers}),
                ("predictions", self.cfg.predictions_every_min,
                 lambda: self.odds_fn(watch, self.cfg.polymarket_queries, self.cfg.kalshi_series)))
        health = self.store.get("health") or {}
        for key, every, fn in plan:
            if _age_min(self.store.get_with_time(key)[1]) < every:
                continue
            try:
                self.store.put(key, fn())
                done[key] = "ok"
                health[f"signal:{key}"] = {"ok_at": utcnow(), "error": None}
            except Exception as e:
                done[key] = f"{type(e).__name__}: {e}"[:200]
                health[f"signal:{key}"] = {**health.get(f"signal:{key}", {}), "error": done[key], "error_at": utcnow()}
        self.store.put("health", health)
        return done

    # -- the loop -------------------------------------------------------------------------------
    def run_forever(self) -> None:
        if self.state() == "stopped":
            self.set_state("running")
        while True:
            t0 = time.time()
            try:
                summary = self.tick()
            except Exception as e:      # one bad tick must not kill a 24/7 service
                summary = {"state": "error", "error": f"{type(e).__name__}: {e}"[:300]}
                self.store.put("last_error", {**summary, "at": utcnow()})
            if summary.get("state") == "stopped":
                self.beat("stopped")
                return
            while time.time() - t0 < self.cfg.interval_s:
                time.sleep(2)
                if self.state() == "stopped":
                    self.beat("stopped")
                    return


CN_NAMES = {"NVDA": ["英伟达"], "AAPL": ["苹果"], "MSFT": ["微软"], "AMZN": ["亚马逊"], "GOOGL": ["谷歌", "Alphabet"],
            "GOOG": ["谷歌", "Alphabet"], "META": ["Meta", "脸书"], "TSLA": ["特斯拉"], "MU": ["美光"], "AMD": ["超威", "AMD"],
            "AVGO": ["博通"], "TSM": ["台积电"], "INTC": ["英特尔"], "QCOM": ["高通"], "ASML": ["阿斯麦"], "LRCX": ["泛林"],
            "AMAT": ["应用材料"], "SMCI": ["超微电脑"], "ORCL": ["甲骨文"], "NFLX": ["奈飞"], "BABA": ["阿里巴巴"],
            "PDD": ["拼多多"], "JPM": ["摩根大通"], "BRK-B": ["伯克希尔"], "XOM": ["埃克森美孚"], "COIN": ["Coinbase"],
            "PLTR": ["Palantir"], "ARM": ["Arm"], "CRWV": ["CoreWeave"], "SPY": ["标普500"], "QQQ": ["纳指100"]}


def default_aliases(t: str) -> list[str]:
    """The company's short name from Yahoo plus its common Chinese name, so headlines in either language match."""
    aliases = []
    try:
        import yfinance as yf
        info = yf.Ticker(t).info or {}
        if info.get("quoteType") == "EQUITY":          # an ETF's long name never appears in headlines
            from equitydesk.tools.news import SUFFIXES
            name = re.sub(r"\.com\b", "", info.get("shortName") or info.get("longName") or "", flags=re.I)
            base = " ".join(SUFFIXES.sub(" ", name).split()).strip(" ,.")
            if base and base.upper() != t:
                aliases.append(base)
    except Exception:
        pass
    return list(dict.fromkeys(aliases + CN_NAMES.get(t, [])))


def seed_watch(store: MonitorStore, tickers: list[str]) -> None:
    """First run: copy the desk's watchlist, with default aliases."""
    if store.watch():
        return
    for t in tickers:
        store.upsert_watch(t, aliases=default_aliases(t))


# -- process management (used by the UI and the launcher) --------------------------------------------

def pid_alive(pid: int | None) -> bool:
    """Is that process still there? A recent heartbeat from a killed process must not read as running."""
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def status(store: MonitorStore, interval_s: int) -> dict:
    """running / paused / stopped / not running, from the heartbeat's age and the control state."""
    hb, at = store.get_with_time("heartbeat")
    age_s = _age_min(at) * 60
    alive = hb is not None and age_s < max(3 * interval_s, 120) and pid_alive((hb or {}).get("pid"))
    state = (hb or {}).get("state", "stopped") if alive else "not running"
    last, last_at = store.get_with_time("last_collect")
    return {"state": state, "alive": alive, "heartbeat_age_s": age_s, "last_collect": last,
            "last_collect_age_s": _age_min(last_at) * 60, "pid": (hb or {}).get("pid")}


SPAWN_PS = r"""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $env:ED_CMD; CurrentDirectory = $env:ED_CWD }
if ($r.ReturnValue -ne 0) { exit [int]$r.ReturnValue }
$r.ProcessId
"""


def spawn(log_path: Path) -> int:
    """Start the service in the background, independent of whoever started it; returns its pid.

    Live finding: started as a child of the Streamlit server, the monitor died whenever the server was restarted
    (a process-tree kill follows parent ids, and a job can forbid breakaway). On Windows the process is therefore
    created by WMI (Win32_Process.Create): its parent is the WMI service, so it belongs to no app's tree or job.
    It runs windowless (pythonw) and writes its own log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        pyw = Path(sys.executable).with_name("pythonw.exe")
        exe = pyw if pyw.exists() else Path(sys.executable)
        env = {**os.environ, "ED_CMD": f'"{exe}" -m equitydesk.monitor --log "{log_path}"', "ED_CWD": str(ROOT)}
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", SPAWN_PS], env=env,
                           capture_output=True, text=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not r.stdout.strip().isdigit():
            raise RuntimeError(f"could not start the monitor: {(r.stderr or r.stdout).strip()[:300]}")
        return int(r.stdout.strip())
    log = open(log_path, "a", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-m", "equitydesk.monitor"], cwd=ROOT, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    return p.pid


STARTUP_NAME = "EquityDesk Monitor.vbs"


def startup_dir() -> Path:
    return Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def autostart_installed() -> bool:
    return sys.platform == "win32" and (startup_dir() / STARTUP_NAME).exists()


def install_autostart() -> Path:
    """Start the monitor, hidden, whenever you log in to Windows (a .vbs in your Startup folder)."""
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = pyw if pyw.exists() else Path(sys.executable)
    script = (f'Set sh = CreateObject("WScript.Shell")\r\nsh.CurrentDirectory = "{ROOT}"\r\n'
              f'sh.Run """{exe}"" -m equitydesk.monitor --log ""{ROOT / "data" / "monitor.log"}""", 0, False\r\n')
    path = startup_dir() / STARTUP_NAME
    path.write_text(script, encoding="utf-8")
    return path


def remove_autostart() -> None:
    (startup_dir() / STARTUP_NAME).unlink(missing_ok=True)


class SingleInstance:
    """A lock file so two monitors never run at once (double-clicking the launcher twice)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise RuntimeError("another monitor is already running")
        return self

    def __exit__(self, *exc):
        try:
            if sys.platform == "win32":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.fh.close()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m equitydesk.monitor", description="EquityDesk market monitor")
    ap.add_argument("--once", action="store_true", help="run one tick and print its summary")
    ap.add_argument("--log", help="append output to this file (used when started in the background)")
    ap.add_argument("--background", action="store_true",
                    help="start the monitor as an independent background process and return (the launcher uses this)")
    a = ap.parse_args(argv)
    if a.log:
        fh = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = fh
    if a.background:
        load_env()
        s = load_settings()
        st = MonitorStore(s.monitor_path)
        if status(st, s.monitor.interval_s)["alive"]:
            print("The monitor is already running.")
            return 0
        st.put("control", {"state": "running"})
        print(f"Monitor started in the background (pid {spawn(s.data_dir / 'monitor.log')}); log: data/monitor.log")
        return 0
    load_env()
    s = load_settings()
    store = MonitorStore(s.monitor_path)
    from equitydesk.memory.store import Store
    seed_watch(store, Store(s.store_path).get_profile()["watchlist"])
    svc = MonitorService(s, store)
    if a.once:
        import json
        print(json.dumps(svc.tick(), ensure_ascii=False, indent=1, default=str))
        return 0
    try:
        with SingleInstance(s.data_dir / "monitor.lock"):
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} monitor started (every {s.monitor.interval_s}s); stop it from "
                  "the Monitor page or with Ctrl+C", flush=True)
            svc.run_forever()
    except RuntimeError as e:
        print(f"Not started: {e}.", flush=True)
        return 1
    except KeyboardInterrupt:
        svc.beat("stopped")
    return 0
