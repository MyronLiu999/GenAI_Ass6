"""Shared fixtures. Every test is offline and free: synthetic data in a temp dir, the scripted model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equitydesk.config import load_settings  # noqa: E402
from tests.synthetic import main as make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixtures")
    make_synthetic(d)
    return d


@pytest.fixture
def settings(tmp_path, fixtures_dir):
    return load_settings(overrides={
        "model": {"provider": "scripted", "name": "scripted", "max_retries": 2},
        "roles": {},
        "data": {"provider": "fixtures", "fixtures_dir": str(fixtures_dir)},
        "paths": {"data_dir": str(tmp_path / "data")},
    })


@pytest.fixture
def make_desk(settings):
    from equitydesk.harness.llm import ScriptedBackend
    from equitydesk.runner import Desk

    desks = []

    def _make(backend: ScriptedBackend | None = None, settings_=None):
        b = backend or ScriptedBackend()
        d = Desk(settings_ or settings, backend_factory=lambda cfg: b)
        desks.append(d)
        return d

    yield _make
    for d in desks:
        d.close()
