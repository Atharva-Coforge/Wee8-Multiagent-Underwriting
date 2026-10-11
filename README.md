# Wee8-Multiagent-Underwriting

## Running tests

`uv run pytest -q` runs everything offline. Live tests are skipped.

`RUN_LIVE_OLLAMA=1 uv run pytest -q -m live` runs the live smoke test with Ollama running.

CI runs the offline suite on every push and pull request.