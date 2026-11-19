FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" FASTEMBED_CACHE_PATH=/models
# Bake the embedding model into the image so the worker starts without a download.
RUN python -c "from rolecall import embed; embed.embed(['warm up'])"
CMD ["rolecall", "ingest"]
