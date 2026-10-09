"""One JSON completion, with a single repair call."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from underwriting.llm_adapter import LLMAdapter, LLMResponse
from underwriting.models import TokenUsage

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


class AgentParseError(Exception):
    """The model did not return schema-valid JSON after one repair call."""

    def __init__(
        self,
        agent: str,
        validation_error: str,
        *,
        usage: TokenUsage | None = None,
        llm_calls: int = 0,
    ) -> None:
        self.agent = agent
        self.validation_error = validation_error
        self.usage = usage
        self.llm_calls = llm_calls
        super().__init__(f"{agent} JSON parse failed: {validation_error}")


class _ReplyInvalid(Exception):
    def __init__(self, validation_error: str) -> None:
        self.validation_error = validation_error
        super().__init__(validation_error)


def call_json(
    llm: LLMAdapter,
    *,
    agent: str,
    system: str,
    user: str,
    schema: type[T],
) -> tuple[T, list[LLMResponse]]:
    """Parse one model reply as ``schema``, repairing invalid JSON at most once.

    Returns the parsed object and every ``LLMResponse`` from this attempt.
    ``LLMTimeoutError`` is not caught and does not start a repair call.
    """
    responses: list[LLMResponse] = []
    first = llm.complete(system=system, user=user)
    responses.append(first)
    try:
        return _parse(first.text, schema), responses
    except _ReplyInvalid as exc:
        error_text = exc.validation_error

    repair = llm.complete(system=system, user=_repair_user(user, first.text, error_text))
    responses.append(repair)
    try:
        parsed = _parse(repair.text, schema)
    except _ReplyInvalid as exc:
        raise AgentParseError(
            agent,
            exc.validation_error,
            usage=usage_from(responses),
            llm_calls=len(responses),
        ) from exc
    return parsed, responses


def usage_from(responses: list[LLMResponse]) -> TokenUsage:
    """Sum token counts across every model call, including a repair."""
    prompt_tokens = sum(response.prompt_tokens for response in responses)
    completion_tokens = sum(response.completion_tokens for response in responses)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _repair_user(original_user: str, bad_reply: str, validation_error: str) -> str:
    return (
        f"{original_user}\n\n"
        "The previous reply was invalid.\n"
        f"Validation error: {validation_error}\n"
        "Previous reply:\n"
        f"{bad_reply}\n"
        "Return only corrected JSON."
    )


def _parse(text: str, schema: type[T]) -> T:
    extracted = _extract_json(text)
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise _ReplyInvalid(str(exc)) from exc
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise _ReplyInvalid(str(exc)) from exc


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = _FENCE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    start = _first_container(stripped)
    if start is None:
        return stripped
    end = _last_container(stripped)
    if end is None or end < start:
        return stripped
    return stripped[start : end + 1]


def _first_container(text: str) -> int | None:
    for index, char in enumerate(text):
        if char in "{[":
            return index
    return None


def _last_container(text: str) -> int | None:
    for index in range(len(text) - 1, -1, -1):
        if text[index] in "}]":
            return index
    return None
