"""Retry a call on ``LLMTimeoutError``, then let the caller escalate."""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

from underwriting.llm_adapter import LLMTimeoutError

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = [1, 2]

T = TypeVar("T")


@dataclass
class Attempt(Generic[T]):
    """One try. ``error`` is set when this try timed out."""

    attempt: int
    started_at: str
    duration_ms: float
    value: T | None = None
    error: LLMTimeoutError | None = None


def retry_on_timeout(
    call: Callable[[int], T],
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_seconds: Sequence[float] = tuple(BACKOFF_SECONDS),
) -> Iterator[Attempt[T]]:
    """Call ``call`` up to ``max_attempts`` times, yielding one :class:`Attempt` each time.

    ``call`` receives the attempt number, starting at 1. Only ``LLMTimeoutError``
    is retried. Any other exception propagates unchanged and is not retried.
    After a timeout, and only when another attempt remains, ``sleep`` is called
    with the next value from ``backoff_seconds``.
    """
    delays = tuple(backoff_seconds)
    for number in range(1, max_attempts + 1):
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        try:
            value = call(number)
        except LLMTimeoutError as exc:
            yield Attempt(
                attempt=number,
                started_at=started_at,
                duration_ms=_elapsed_ms(started),
                error=exc,
            )
            if number == max_attempts:
                return
            sleep(delays[number - 1])
            continue
        yield Attempt(
            attempt=number,
            started_at=started_at,
            duration_ms=_elapsed_ms(started),
            value=value,
        )
        return


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
