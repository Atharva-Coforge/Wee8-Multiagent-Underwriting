"""An agent's output plus the tokens its model calls used."""

from typing import Generic, TypeVar

from pydantic import BaseModel

from underwriting.models import TokenUsage

T = TypeVar("T")


class AgentResult(BaseModel, Generic[T]):
    output: T
    usage: TokenUsage
    llm_calls: int
