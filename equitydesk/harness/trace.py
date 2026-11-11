"""Observability: one JSONL trace per run, one line per node span, model call, tool call and guard event.

data/runs/<run_id>.jsonl          the trace
data/runs/<run_id>.prompts.jsonl  every prompt sent to a model, numbered (debugging and audits)
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Tracer:
    def __init__(self, run_id: str, runs_dir: Path):
        self.run_id = run_id
        self.dir = Path(runs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"
        self.prompts_path = self.dir / f"{run_id}.prompts.jsonl"
        self._lock = threading.Lock()
        self._request_no = self._last_request_no()

    def _last_request_no(self) -> int:
        """A resumed run continues its prompt numbering."""
        if not self.prompts_path.exists():
            return 0
        n = 0
        for line in self.prompts_path.read_text(encoding="utf-8").splitlines():
            try:
                n = max(n, json.loads(line).get("request", 0))
            except json.JSONDecodeError:
                continue
        return n

    def _write(self, path: Path, rec: dict) -> None:
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def event(self, type_: str, **fields: Any) -> None:
        self._write(self.path, {"ts": now_iso(), "t": round(time.time(), 3), "run_id": self.run_id, "type": type_,
                                **fields})

    def prompt(self, node: str, model: str, messages: list[dict]) -> int:
        with self._lock:
            self._request_no += 1
            n = self._request_no
        self._write(self.prompts_path, {"request": n, "ts": now_iso(), "node": node, "model": model,
                                        "messages": messages})
        return n

    @contextmanager
    def span(self, node: str, **fields: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        self.event("node_start", node=node, **fields)
        try:
            yield
        except Exception as e:
            self.event("node_end", node=node, ok=False, ms=round((time.perf_counter() - t0) * 1000),
                       error=f"{type(e).__name__}: {e}"[:500])
            raise
        self.event("node_end", node=node, ok=True, ms=round((time.perf_counter() - t0) * 1000))


def read_trace(run_id: str, runs_dir: Path) -> list[dict]:
    p = Path(runs_dir) / f"{run_id}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_prompts(run_id: str, runs_dir: Path) -> list[dict]:
    p = Path(runs_dir) / f"{run_id}.prompts.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
