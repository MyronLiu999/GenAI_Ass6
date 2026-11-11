"""The long-term store (SQLite). One file, four tables: runs, analyses, notes, profile.

Analyses are append-only records of what the desk said and the price when it said it; that is what
makes the track record honest. Notes are the user's own statements, kept with their provenance and
soft-deleted (inactive), never edited in place.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL, model TEXT, debate_rounds INTEGER, tokens INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0,
    error TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, ticker TEXT NOT NULL, created_at TEXT NOT NULL, as_of TEXT,
    rating TEXT NOT NULL, score INTEGER NOT NULL, conviction INTEGER, horizon_months INTEGER,
    price REAL NOT NULL, benchmark TEXT, benchmark_price REAL, thesis TEXT, grounded INTEGER, report_json TEXT,
    active INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS ix_analyses_ticker ON analyses(ticker, created_at);
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY, ticker TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user', active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

DEFAULT_PROFILE: dict[str, Any] = {
    "portfolio_value": 100_000.0,
    "risk_per_trade_pct": 1.0,
    "max_position_pct": 10.0,
    "risk_tolerance": "moderate",      # conservative | moderate | aggressive
    "horizon_months": 6,
    "watchlist": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"],
}
GLOBAL = "*"   # a note that applies to every ticker


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript(SCHEMA)

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

    # -- runs (the ledger) -----------------------------------------------------------------------
    def start_run(self, run_id: str, ticker: str, model: str, debate_rounds: int, source: str) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO runs (run_id, ticker, started_at, status, model, debate_rounds, source) "
                       "VALUES (?,?,?,?,?,?,?)", (run_id, ticker, now(), "running", model, debate_rounds, source))

    def finish_run(self, run_id: str, status: str, tokens: int, cost_usd: float, error: str | None = None) -> None:
        with self._db() as db:
            db.execute("UPDATE runs SET status=?, finished_at=?, tokens=tokens+?, cost_usd=cost_usd+?, error=? "
                       "WHERE run_id=?", (status, now(), tokens, cost_usd, error, run_id))

    def add_usage(self, run_id: str, tokens: int, cost_usd: float) -> None:
        """Follow-up Q&A spend is charged to the report's run."""
        with self._db() as db:
            db.execute("UPDATE runs SET tokens=tokens+?, cost_usd=cost_usd+? WHERE run_id=?", (tokens, cost_usd, run_id))

    def mark_running(self, run_id: str) -> None:
        with self._db() as db:
            db.execute("UPDATE runs SET status='running', error=NULL WHERE run_id=?", (run_id,))

    def get_run(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 200) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))]

    # -- analyses (episodic memory of past calls) ------------------------------------------------
    def save_analysis(self, run_id: str, ticker: str, as_of: str, decision: dict, price: float, benchmark: str,
                      benchmark_price: float | None, grounded: bool, report: dict) -> str:
        from equitydesk.schemas import RATING_SCORE
        aid = uuid.uuid4().hex[:12]
        with self._db() as db:
            db.execute("UPDATE analyses SET active=0 WHERE run_id=?", (run_id,))  # a resumed run replaces its own
            db.execute(
                "INSERT INTO analyses (id, run_id, ticker, created_at, as_of, rating, score, conviction, horizon_months,"
                " price, benchmark, benchmark_price, thesis, grounded, report_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (aid, run_id, ticker, now(), as_of, decision["rating"], RATING_SCORE[decision["rating"]],
                 decision.get("conviction"), decision.get("horizon_months"), price, benchmark, benchmark_price,
                 decision.get("thesis", ""), int(grounded), json.dumps(report, default=str)))
        return aid

    def list_analyses(self, ticker: str | None = None, limit: int = 500, with_report: bool = False) -> list[dict]:
        cols = "*" if with_report else ("id, run_id, ticker, created_at, as_of, rating, score, conviction, horizon_months,"
                                        " price, benchmark, benchmark_price, thesis, grounded")
        q, args = f"SELECT {cols} FROM analyses WHERE active=1", []
        if ticker:
            q, args = q + " AND ticker=?", [ticker]
        with self._db() as db:
            return [dict(r) for r in db.execute(q + " ORDER BY created_at DESC LIMIT ?", (*args, limit))]

    def get_report(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT report_json FROM analyses WHERE run_id=? AND active=1", (run_id,)).fetchone()
        return json.loads(row["report_json"]) if row and row["report_json"] else None

    def delete_analysis(self, analysis_id: str) -> None:
        with self._db() as db:
            db.execute("UPDATE analyses SET active=0 WHERE id=?", (analysis_id,))

    # -- notes (the user's own statements) -------------------------------------------------------
    def add_note(self, ticker: str, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("empty note")
        nid = uuid.uuid4().hex[:12]
        with self._db() as db:
            db.execute("INSERT INTO notes (id, ticker, text, created_at) VALUES (?,?,?,?)",
                       (nid, ticker.upper() if ticker != GLOBAL else GLOBAL, text[:1000], now()))
        return nid

    def list_notes(self, ticker: str | None = None, include_global: bool = True) -> list[dict]:
        q = "SELECT * FROM notes WHERE active=1"
        args: list[Any] = []
        if ticker:
            q += " AND (ticker=?" + (" OR ticker='*')" if include_global else ")")
            args.append(ticker)
        with self._db() as db:
            return [dict(r) for r in db.execute(q + " ORDER BY created_at DESC", args)]

    def deactivate_note(self, note_id: str) -> None:
        with self._db() as db:
            db.execute("UPDATE notes SET active=0 WHERE id=?", (note_id,))

    # -- profile (semantic memory about the user) ------------------------------------------------
    def get_profile(self) -> dict:
        with self._db() as db:
            rows = {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT key, value FROM profile")}
        return {**DEFAULT_PROFILE, **rows}

    def set_profile(self, values: dict) -> None:
        unknown = set(values) - set(DEFAULT_PROFILE)
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        with self._db() as db:
            for k, v in values.items():
                db.execute("INSERT OR REPLACE INTO profile (key, value) VALUES (?,?)", (k, json.dumps(v)))
