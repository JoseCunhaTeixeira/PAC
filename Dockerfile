# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
WORKDIR /app

# git: needed by `uv sync` to clone the sigpipe dependency
# (sigpipe @ git+https://github.com/JoseCunhaTeixeira/sigpipe).
# build-essential: bayesbay (sigpipe's MCMC dependency) ships no prebuilt
# wheel for Python 3.14 yet -- uv compiles its bayesbay._utils_1d C++
# extension from source on install, which needs g++.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

FROM python:3.14-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user matching the typical first Linux user (uid/gid 1000)
# so files written into the bind-mounted data/input and data/output (e.g.
# computation results) are owned by the host user, not root -- override
# with --build-arg APP_UID/APP_GID if your host user differs.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" app && useradd -u "${APP_UID}" -g app -M app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app
# WORKDIR above creates /app as root before the COPY runs, so --chown only
# covers the copied contents, not the /app directory entry itself -- sigpipe
# writes a relative "logs" dir from the current working directory at
# runtime, which needs the cwd itself to be writable by `app`.
RUN chown app:app /app
ENV PATH="/app/.venv/bin:$PATH"
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "masw.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
