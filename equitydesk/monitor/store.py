"""The monitor's SQLite store. WAL mode: the background service writes while the UI reads."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    key TEXT PRIMARY KEY, source_key TEXT NOT NULL, source TEXT NOT NULL, tier TEXT NOT NULL,
    title TEXT NOT NULL, url TEXT, published TEXT NOT NULL, summary TEXT, fetched_at TEXT NOT NULL,
    hint_tickers TEXT, flagged INTEGER DEFAULT 0,
    cluster TEXT,                -- key of the first item of the same story; NULL for the head itself
    n_sources INTEGER DEFAULT 1, -- heads only: how many sources carried the story
    score INTEGER, category TEXT, tickers TEXT, title_t TEXT, why TEXT, rumor INTEGER, scored_at TEXT,
    pushed_at TEXT, report_id TEXT);
CREATE INDEX IF NOT EXISTS ix_items_pub ON items(published);
CREATE INDEX IF NOT EXISTS ix_items_score ON items(scored_at, score);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, item_key TEXT NOT NULL, created_at TEXT NOT NULL, title TEXT, body TEXT,
    sources TEXT, cost_usd REAL);
CREATE TABLE IF NOT EXISTS watch (
    ticker TEXT PRIMARY KEY, aliases TEXT NOT NULL DEFAULT '[]', match_ticker INTEGER NOT NULL DEFAULT 1,
    holding INTEGER NOT NULL DEFAULT 0, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
"""

ITEM_JSON = ("hint_tickers", "tickers")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_local() -> str:
    return time.strftime("%Y-%m-%d")


class MonitorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            cols = {r[1] for r in db.execute("PRAGMA table_info(items)")}
            if "push_sent" not in cols:      # added after the first release
                db.execute("ALTER TABLE items ADD COLUMN push_sent INTEGER DEFAULT 0")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                yield db
                db.commit()
            finally:
                db.close()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ITEM_JSON:
            if k in d:
                d[k] = json.loads(d[k]) if d[k] else []
        return d

    # -- key/value state (control, heartbeat, quotes, signals, overview, health, spend) ----------
    def get(self, key: str, default: Any = None) -> Any:
        with self._db() as db:
            r = db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def get_with_time(self, key: str) -> tuple[Any, str | None]:
        with self._db() as db:
            r = db.execute("SELECT value, updated_at FROM kv WHERE key=?", (key,)).fetchone()
        return (json.loads(r["value"]), r["updated_at"]) if r else (None, None)

    def put(self, key: str, value: Any) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?,?,?)",
                       (key, json.dumps(value, ensure_ascii=False, default=str), utcnow()))

    def add_spend(self, usd: float) -> float:
        key = f"spend:{today_local()}"
        total = float(self.get(key, 0.0)) + usd
        self.put(key, total)
        return total

    def spend_today(self) -> float:
        return float(self.get(f"spend:{today_local()}", 0.0))

    # -- items ----------------------------------------------------------------------------------
    def add_items(self, items: list[dict]) -> list[dict]:
        """Insert new items (by key); returns those that were new."""
        new = []
        with self._db() as db:
            for it in items:
                cur = db.execute(
                    "INSERT OR IGNORE INTO items (key, source_key, source, tier, title, url, published, summary, "
                    "fetched_at, hint_tickers, flagged) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (it["key"], it["source_key"], it["source"], it["tier"], it["title"], it.get("url", ""),
                     it["published"], it.get("summary", ""), utcnow(), json.dumps(it.get("hint_tickers") or []),
                     int(bool(it.get("flagged")))))
                if cur.rowcount:
                    new.append(it)
        return new

    def set_cluster(self, key: str, head: str) -> None:
        with self._db() as db:
            db.execute("UPDATE items SET cluster=? WHERE key=?", (head, key))
            db.execute("UPDATE items SET n_sources = n_sources + 1 WHERE key=?", (head,))

    def set_score(self, key: str, score: int, category: str, tickers: list[str], title_t: str, why: str,
                  rumor: bool) -> None:
        with self._db() as db:
            db.execute("UPDATE items SET score=?, category=?, tickers=?, title_t=?, why=?, rumor=?, scored_at=? "
                       "WHERE key=?", (int(score), category, json.dumps(tickers), title_t, why, int(rumor), utcnow(), key))

    def mark_pushed(self, key: str, sent: bool = True) -> None:
        """Handled once: sent, or held back (hourly cap, first-tick backlog), never retried later."""
        with self._db() as db:
            db.execute("UPDATE items SET pushed_at=?, push_sent=? WHERE key=?", (utcnow(), int(sent), key))

    def recent_heads(self, hours: float = 24) -> list[dict]:
        with self._db() as db:
            return [self._row(r) for r in db.execute(
                "SELECT * FROM items WHERE cluster IS NULL AND published >= ? ORDER BY published DESC", (ago(hours),))]

    def unscored(self, limit: int = 200) -> list[dict]:
        with self._db() as db:
            return [self._row(r) for r in db.execute(
                "SELECT * FROM items WHERE cluster IS NULL AND scored_at IS NULL ORDER BY published DESC LIMIT ?",
                (limit,))]

    def items(self, hours: float = 24, min_score: int | None = None, heads_only: bool = True,
              limit: int = 2000) -> list[dict]:
        q, args = "SELECT * FROM items WHERE published >= ?", [ago(hours)]
        if heads_only:
            q += " AND cluster IS NULL"
        if min_score is not None:
            q += " AND score >= ?"
            args.append(min_score)
        with self._db() as db:
            return [self._row(r) for r in db.execute(q + " ORDER BY published DESC LIMIT ?", (*args, limit))]

    def cluster_members(self, head: str) -> list[dict]:
        with self._db() as db:
            return [self._row(r) for r in db.execute("SELECT * FROM items WHERE cluster=? ORDER BY published", (head,))]

    def get_item(self, key: str) -> dict | None:
        with self._db() as db:
            r = db.execute("SELECT * FROM items WHERE key=?", (key,)).fetchone()
        return self._row(r) if r else None

    def to_push(self, push_min: int, holding_min: int, holdings: set[str]) -> list[dict]:
        with self._db() as db:
            rows = [self._row(r) for r in db.execute(
                "SELECT * FROM items WHERE cluster IS NULL AND scored_at IS NOT NULL AND pushed_at IS NULL "
                "AND score >= ? AND published >= ?", (min(push_min, holding_min), ago(2)))]
        return [r for r in rows if r["score"] >= push_min or (set(r["tickers"]) & holdings and r["score"] >= holding_min)]

    def counts_today(self) -> dict:
        start = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0) \
            .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._db() as db:
            r = db.execute(
                "SELECT COUNT(*) n, SUM(scored_at IS NOT NULL) scored, SUM(score >= 8) high, SUM(push_sent = 1) "
                "pushed FROM items WHERE fetched_at >= ? AND cluster IS NULL", (start,)).fetchone()
            reports = db.execute("SELECT COUNT(*) FROM reports WHERE created_at >= ?", (start,)).fetchone()[0]
        return {"items": r["n"] or 0, "scored": r["scored"] or 0, "high": r["high"] or 0, "pushed": r["pushed"] or 0,
                "reports": reports}

    def prune(self, keep_days: int) -> int:
        cut = ago(keep_days * 24)
        with self._db() as db:
            n = db.execute("DELETE FROM items WHERE published < ? AND report_id IS NULL", (cut,)).rowcount
        return n

    # -- reports --------------------------------------------------------------------------------
    def add_report(self, report_id: str, item_key: str, title: str, body: str, sources: list[dict],
                   cost: float) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO reports (id, item_key, created_at, title, body, sources, cost_usd) "
                       "VALUES (?,?,?,?,?,?,?)", (report_id, item_key, utcnow(), title, body,
                                                  json.dumps(sources, ensure_ascii=False), cost))
            db.execute("UPDATE items SET report_id=? WHERE key=?", (report_id, item_key))

    def reports(self, limit: int = 50) -> list[dict]:
        with self._db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,))]
        for r in rows:
            r["sources"] = json.loads(r["sources"] or "[]")
        return rows

    def get_report(self, report_id: str) -> dict | None:
        with self._db() as db:
            r = db.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["sources"] = json.loads(d["sources"] or "[]")
        return d

    def reports_today(self) -> int:
        return self.counts_today()["reports"]

    # -- watchlist (aliases, match-by-ticker, holding) -------------------------------------------
    def watch(self) -> list[dict]:
        with self._db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM watch ORDER BY added_at, ticker")]
        for r in rows:
            r["aliases"] = json.loads(r["aliases"] or "[]")
            r["match_ticker"], r["holding"] = bool(r["match_ticker"]), bool(r["holding"])
        return rows

    def upsert_watch(self, ticker: str, aliases: list[str] | None = None, match_ticker: bool | None = None,
                     holding: bool | None = None) -> None:
        cur = {w["ticker"]: w for w in self.watch()}.get(ticker)
        aliases = aliases if aliases is not None else (cur["aliases"] if cur else [])
        match_ticker = match_ticker if match_ticker is not None else (cur["match_ticker"] if cur else len(ticker) >= 2)
        holding = holding if holding is not None else (cur["holding"] if cur else False)
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO watch (ticker, aliases, match_ticker, holding, added_at) VALUES "
                       "(?,?,?,?,COALESCE((SELECT added_at FROM watch WHERE ticker=?), ?))",
                       (ticker, json.dumps([a.strip() for a in aliases if a.strip()], ensure_ascii=False),
                        int(match_ticker), int(holding), ticker, utcnow()))

    def remove_watch(self, ticker: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM watch WHERE ticker=?", (ticker,))
