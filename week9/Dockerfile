# syntax=docker/dockerfile:1
# Risk scoring agent image. Build from the repo root (Graviton / ARM64):
#   docker build --platform linux/arm64 -f week9/Dockerfile -t risk-scoring-agent:v1 .
#
# Copies src/underwriting and week9/risk_agent only, plus the fixtures used as
# test input. No data/ (applicant databases), tests/, or traces/.

# --- Build stage: dependencies only -----------------------------------------
FROM python:3.12.14-slim-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Week 8 runtime dependencies at the exact versions in uv.lock, without dev
# tools. boto3 is pinned here because the root pyproject.toml stays unchanged.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
    && uv pip install --python /opt/venv/bin/python \
        boto3==1.43.114 botocore==1.43.114

# --- Runtime stage ------------------------------------------------------------
FROM python:3.12.14-slim-trixie

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin agent

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY src/underwriting src/underwriting
COPY week9/__init__.py week9/__init__.py
COPY week9/risk_agent week9/risk_agent
COPY week9/fixtures week9/fixtures

# Logs go to stdout unbuffered for CloudWatch. No .pyc writes, so the task can
# run with a read-only root filesystem.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001

ENTRYPOINT ["python", "-m", "week9.risk_agent.run_risk_agent"]
CMD ["week9/fixtures/maria_ortiz.json", \
     "week9/fixtures/tyler_brandt.json", \
     "week9/fixtures/clean_claim_mismatch.json", \
     "week9/fixtures/messy_formats.json", \
     "week9/fixtures/timeout_case.json"]
