"""Ollama call, token counts, and timeout mapping."""

import os
from typing import Any, Protocol

import httpx
from ollama import Client
from pydantic import BaseModel

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_TIMEOUT_SECONDS = 120.0


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMTimeoutError(TimeoutError):
    """The model client timed out."""


class LLMAdapter(Protocol):
    def complete(self, *, system: str, user: str) -> LLMResponse: ...


class OllamaAdapter:
    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self._client = client or Client(
            host=host or os.getenv("OLLAMA_HOST", DEFAULT_HOST),
            timeout=_timeout_seconds(timeout),
        )

    def complete(self, *, system: str, user: str) -> LLMResponse:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                think=False,
                format="json",
                options={"temperature": 0.1},
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc

        message = response.message
        text = "" if message is None or message.content is None else message.content
        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _timeout_seconds(timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    raw = os.getenv("OLLAMA_TIMEOUT")
    if raw:
        return float(raw)
    return DEFAULT_TIMEOUT_SECONDS
