FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    anthropic>=0.28 \
    openai>=1.30 \
    prometheus-client>=0.20 \
    psycopg2-binary>=2.9.9 \
    pgvector>=0.2.5 \
    pydantic>=2.7 \
    pydantic-settings>=2.2 \
    python-dotenv>=1.0 \
    httpx>=0.27

COPY api/ ./api/
COPY embeddings/ ./embeddings/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
