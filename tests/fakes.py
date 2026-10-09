"""Fake LLM adapter shared by the tests.

Scripts replies, records every call, and never starts Ollama.
"""

import json
from dataclasses import dataclass
from typing import Any

from underwriting.llm_adapter import LLMResponse, LLMTimeoutError

ScriptedReply = dict[str, Any] | str | BaseException | type[BaseException]

_INVALID_JSON = '{"score": 72,'


@dataclass(frozen=True)
class RecordedCall:
    """One ``complete`` call, including calls that raise ``LLMTimeoutError``."""

    system: str
    user: str
    index: int


class FakeLLMAdapter:
    """LLM adapter that returns a fixed script and records each call.

    Each scripted reply is a ``dict`` (serialized as JSON), a ``str`` (returned
    unchanged), or an exception class or instance (raised). When the script is
    exhausted, the next call raises ``AssertionError``.
    """

    def __init__(
        self,
        replies: list[ScriptedReply],
        *,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> None:
        self._replies = list(replies)
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[RecordedCall] = []

    @classmethod
    def bad_json_then(cls, reply: ScriptedReply) -> "FakeLLMAdapter":
        """First reply is invalid JSON; the second is ``reply``."""
        return cls([_INVALID_JSON, reply])

    @classmethod
    def timeouts_then(
        cls,
        n: int,
        reply: ScriptedReply | None = None,
    ) -> "FakeLLMAdapter":
        """Raise ``LLMTimeoutError`` on the first ``n`` calls, then return ``reply``.

        With ``reply=None``, the script is only those ``n`` timeouts.
        """
        script: list[ScriptedReply] = [LLMTimeoutError] * n
        if reply is not None:
            script.append(reply)
        return cls(script)

    @property
    def call_count(self) -> int:
        """How many times ``complete`` has been called."""
        return len(self.calls)

    def complete(self, *, system: str, user: str) -> LLMResponse:
        """Return the next scripted reply, or raise if the script says to."""
        index = self.call_count + 1
        self.calls.append(RecordedCall(system=system, user=user, index=index))
        scripted = len(self._replies)
        if index > scripted:
            raise AssertionError(
                f"FakeLLMAdapter scripted {scripted} calls; another call was made."
            )
        reply = self._replies[index - 1]
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, type) and issubclass(reply, BaseException):
            raise reply()
        if isinstance(reply, str):
            text = reply
        elif isinstance(reply, dict):
            text = json.dumps(reply)
        else:
            raise TypeError(
                "Scripted reply must be a dict, str, or exception, "
                f"got {type(reply).__name__}."
            )
        return LLMResponse(
            text=text,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )
