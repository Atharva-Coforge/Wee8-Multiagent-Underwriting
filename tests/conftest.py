"""Shared pytest hooks."""

import os

import pytest

_LIVE_SKIP = pytest.mark.skip(
    reason="needs a running local Ollama; skipped unless RUN_LIVE_OLLAMA=1"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip live Ollama tests unless RUN_LIVE_OLLAMA=1."""
    if os.environ.get("RUN_LIVE_OLLAMA") == "1":
        return
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(_LIVE_SKIP)
