"""Bedrock Converse call for Gemma 3 27B, token counts, and timeout mapping.

Implements the week 8 ``LLMAdapter`` protocol, so no agent code changes.
Credentials come from the ECS task role. No API keys.

``boto3`` is imported only when no client is passed in, so the tests run with a
fake client and without ``boto3`` installed.
"""

import os
from typing import Any

from underwriting.llm_adapter import LLMResponse, LLMTimeoutError

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "google.gemma-3-27b-it"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 1024
CONNECT_TIMEOUT_SECONDS = 5.0


class BedrockAdapter:
    """One Converse call per ``complete``. Temperature 0, like the Ollama adapter.

    Gemma has no native system role. If Bedrock rejects the ``system`` field for
    it, set ``system_in_user`` (or ``BEDROCK_SYSTEM_IN_USER=1``) to send the
    system prompt at the top of the user message instead.

    ``timeout_errors`` are the exception types that mean the call timed out.
    With the real client they are botocore's read and connect timeouts.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_in_user: bool | None = None,
        client: Any = None,
        timeout_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        self.model_id: str = (
            model_id or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
        )
        self.max_tokens = max_tokens
        self.system_in_user = (
            system_in_user
            if system_in_user is not None
            else os.getenv("BEDROCK_SYSTEM_IN_USER") == "1"
        )
        if client is None:
            self._client = _bedrock_client(
                region or os.getenv("AWS_REGION") or DEFAULT_REGION,
                _timeout_seconds(timeout),
            )
            self._timeout_errors = _botocore_timeouts()
        else:
            self._client = client
            self._timeout_errors = timeout_errors

    def complete(self, *, system: str, user: str) -> LLMResponse:
        try:
            response = self._client.converse(**self._request(system, user))
        except self._timeout_errors as exc:
            raise LLMTimeoutError(str(exc)) from exc

        usage = response.get("usage", {})
        return LLMResponse(
            text=_text(response),
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
        )

    def _request(self, system: str, user: str) -> dict[str, Any]:
        text = f"{system}\n\n{user}" if self.system_in_user else user
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": text}]}],
            "inferenceConfig": {"temperature": 0, "maxTokens": self.max_tokens},
        }
        if not self.system_in_user:
            request["system"] = [{"text": system}]
        return request


def _text(response: dict[str, Any]) -> str:
    """Join the text blocks of the reply. No text gives "", which the JSON repair handles."""
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(block["text"] for block in content if "text" in block)


def _bedrock_client(region: str, timeout: float) -> Any:
    """bedrock-runtime client with one attempt per call.

    botocore retries are off so the week 8 ``retry_on_timeout`` is the only
    retry policy. Otherwise each of its 3 attempts could retry again inside
    botocore, and the worst-case latency would no longer be 3 x the timeout.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            read_timeout=timeout,
            retries={"total_max_attempts": 1},
        ),
    )


def _botocore_timeouts() -> tuple[type[BaseException], ...]:
    from botocore.exceptions import ConnectTimeoutError, ReadTimeoutError

    return (ReadTimeoutError, ConnectTimeoutError)


def _timeout_seconds(timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    raw = os.getenv("BEDROCK_TIMEOUT")
    if raw:
        return float(raw)
    return DEFAULT_TIMEOUT_SECONDS
