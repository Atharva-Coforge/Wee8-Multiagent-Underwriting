"""Retry only LLMTimeoutError, with a fake sleep."""

import pytest

from underwriting.llm_adapter import LLMTimeoutError
from underwriting.pipeline.retry import BACKOFF_SECONDS, MAX_ATTEMPTS, retry_on_timeout


def test_success_on_first_attempt():
    slept: list[float] = []

    attempts = list(retry_on_timeout(lambda _attempt: "ok", sleep=slept.append))

    assert MAX_ATTEMPTS == 3
    assert len(attempts) == 1
    assert attempts[0].attempt == 1
    assert attempts[0].value == "ok"
    assert attempts[0].error is None
    assert attempts[0].duration_ms >= 0
    assert attempts[0].started_at
    assert slept == []


def test_recovers_on_third_attempt_after_two_timeouts():
    calls = {"n": 0}

    def call(attempt: int) -> str:
        calls["n"] += 1
        assert attempt == calls["n"]
        if calls["n"] < 3:
            raise LLMTimeoutError("timed out")
        return "score"

    slept: list[float] = []
    attempts = list(retry_on_timeout(call, sleep=slept.append))

    assert [item.attempt for item in attempts] == [1, 2, 3]
    assert [item.error is not None for item in attempts] == [True, True, False]
    assert str(attempts[0].error) == "timed out"
    assert str(attempts[1].error) == "timed out"
    assert attempts[2].value == "score"
    assert attempts[2].error is None
    assert slept == [1, 2]


def test_gives_up_after_three_timeouts():
    calls = {"n": 0}

    def call(_attempt: int) -> str:
        calls["n"] += 1
        raise LLMTimeoutError("timed out")

    slept: list[float] = []
    attempts = list(retry_on_timeout(call, sleep=slept.append))

    assert calls["n"] == 3
    assert [item.attempt for item in attempts] == [1, 2, 3]
    assert [str(item.error) for item in attempts] == ["timed out", "timed out", "timed out"]
    assert all(item.value is None for item in attempts)
    assert slept == [1, 2]


def test_non_timeout_error_is_not_retried():
    calls = {"n": 0}
    exc = RuntimeError("bad json")

    def call(_attempt: int) -> str:
        calls["n"] += 1
        raise exc

    slept: list[float] = []
    with pytest.raises(RuntimeError) as caught:
        list(retry_on_timeout(call, sleep=slept.append))

    assert caught.value is exc
    assert calls["n"] == 1
    assert slept == []


def test_backoff_sleeps_with_the_configured_delays():
    calls = {"n": 0}

    def call(_attempt: int) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMTimeoutError("timed out")
        return "score"

    slept: list[float] = []
    list(retry_on_timeout(call, sleep=slept.append))

    assert BACKOFF_SECONDS == [1, 2]
    assert slept == [1, 2]
    assert calls["n"] == 3
