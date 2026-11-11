"""Prices from pricing.yaml. Never hard-code a price anywhere else."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from equitydesk.config import ROOT


@lru_cache(maxsize=4)
def _sheet(path: str) -> dict:
    p = Path(path)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("models", {}) if p.exists() else {}


def price_line(model: str, path: Path | None = None) -> dict | None:
    """The longest pricing line the served model name starts with (dated names match their base)."""
    sheet = _sheet(str(path or ROOT / "pricing.yaml"))
    best = None
    for name, line in sheet.items():
        if model == name or model.startswith(name + "-") or model.startswith(name):
            if best is None or len(name) > len(best[0]):
                best = (name, line)
    return best[1] if best else None


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0,
             path: Path | None = None) -> float | None:
    """USD for one call, or None when the model is unpriced. input_tokens includes cache reads."""
    line = price_line(model, path)
    if not line:
        return None
    uncached = max(input_tokens - cache_read_tokens, 0)
    cr = line.get("cache_read", line["input"])
    return (uncached * line["input"] + cache_read_tokens * cr + output_tokens * line["output"]) / 1_000_000
