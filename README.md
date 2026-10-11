# Wee8-Multiagent-Underwriting

Personal-auto applications go through four agents in a fixed order: intake, enrichment, risk scoring, and recommendation. Each applicant is a plain-text file in `data/applications/`. Enrichment looks that person up in the fictional driving-history, vehicle, and prior-insurance files under `data/databases/`, matching normalized name and date of birth. A completed case ends as `approve`, `deny`, or `refer`. Missing identity information ends as `reject`. Each run writes `traces/<case_id>.jsonl`.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running on the host, with `qwen3.5:9b` pulled

`scripts/run_cases.py` calls Ollama at `http://host.docker.internal:11434`. Set `OLLAMA_MODEL` to use a different model name. The default is `qwen3.5:9b`.

```bash
ollama pull qwen3.5:9b
uv sync
```

## Running the application

Case names are the file stems in `data/applications/`.

```bash
uv run python scripts/run_cases.py maria_ortiz
uv run python scripts/run_cases.py maria_ortiz tyler_brandt
uv run python scripts/run_cases.py --all
uv run python scripts/run_cases.py --inject-timeout 3 timeout_case
```

- `maria_ortiz` runs one application.
- `maria_ortiz tyler_brandt` runs those two, in the order given.
- `--all` runs every `.txt` file in `data/applications/`. Pass case names or `--all`, not both.
- `--inject-timeout 3 timeout_case` raises a timeout on the first three risk-scoring calls, then lets the retry path finish. `timeout_case` is the application meant for that path. `N` can be any count of 0 or greater.

## Running tests

`uv run pytest -q` runs everything offline. Live tests are skipped.

`RUN_LIVE_OLLAMA=1 uv run pytest -q -m live` runs the live smoke test with Ollama running.

CI runs ruff, mypy, and the offline suite on every push and pull request.
