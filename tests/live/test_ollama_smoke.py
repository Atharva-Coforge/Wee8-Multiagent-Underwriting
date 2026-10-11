"""One smoke test against a local Ollama. Skipped unless RUN_LIVE_OLLAMA=1."""

from pathlib import Path

import pytest

from underwriting.llm_adapter import OllamaAdapter
from underwriting.pipeline import run_pipeline

_APPLICATION = (
    Path(__file__).resolve().parents[2] / "data" / "applications" / "maria_ortiz.txt"
)


@pytest.mark.live
def test_maria_ortiz_completes_against_ollama(tmp_path: Path) -> None:
    result = run_pipeline(
        _APPLICATION.read_text(),
        case_id="maria_ortiz",
        llm=OllamaAdapter(),
        trace_dir=tmp_path,
    )

    assert result.status == "completed"
    assert result.decision in ("approve", "deny", "refer")
