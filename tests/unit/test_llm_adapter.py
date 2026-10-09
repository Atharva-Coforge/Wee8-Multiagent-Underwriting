"""Ollama adapter behavior without a running model."""

import httpx

from underwriting.llm_adapter import LLMTimeoutError, OllamaAdapter


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Response:
    def __init__(self, content: str, prompt_eval_count: int, eval_count: int) -> None:
        self.message = _Message(content)
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count


class _Client:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def test_complete_returns_text_and_token_counts():
    client = _Client(_Response('{"ok": true}', 11, 4))
    adapter = OllamaAdapter(model="qwen3.5:9b", client=client)

    result = adapter.complete(system="system", user="user")

    assert result.text == '{"ok": true}'
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 4
    call = client.calls[0]
    assert call["model"] == "qwen3.5:9b"
    assert call["think"] is False
    assert call["format"] == "json"
    assert call["options"]["temperature"] == 0.1
    assert call["messages"][0] == {"role": "system", "content": "system"}
    assert call["messages"][1] == {"role": "user", "content": "user"}


def test_client_timeout_raises_llm_timeout_error():
    client = _Client(error=httpx.ReadTimeout("timed out"))
    adapter = OllamaAdapter(client=client)

    try:
        adapter.complete(system="system", user="user")
    except LLMTimeoutError:
        return
    raise AssertionError("expected LLMTimeoutError")
