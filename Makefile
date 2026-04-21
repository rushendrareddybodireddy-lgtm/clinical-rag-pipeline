.PHONY: help up down restart logs build shell-api shell-postgres \
        init-db verify-mimic run-spark-bronze run-spark-silver run-spark-gold \
        embed-notes test lint format

COMPOSE = docker compose
API_CTR = clinical_api
PG_CTR  = clinical_postgres

help:
	@echo ""
	@echo "  Clinical RAG Pipeline — available targets"
	@echo "  ─────────────────────────────────────────"
	@echo "  up               Start all services"
	@echo "  down             Stop and remove containers"
	@echo "  restart          Restart all services"
	@echo "  logs             Tail logs (all services)"
	@echo "  build            Rebuild images"
	@echo ""
	@echo "  init-db          Run DB migrations (pgvector schema)"
	@echo "  verify-mimic     Check MIMIC-III CSVs are present in data/mimic/"
	@echo ""
	@echo "  run-spark-bronze   Trigger Bronze ingestion job"
	@echo "  run-spark-silver   Trigger Silver transform job"
	@echo "  run-spark-gold     Trigger Gold SOFA scoring job"
	@echo "  embed-notes        Run embedding pipeline manually"
	@echo ""
	@echo "  test             Run test suite"
	@echo "  lint             Run ruff linter"
	@echo "  format           Run ruff formatter"
	@echo ""

up:
	@cp -n .env.example .env 2>/dev/null || true
	$(COMPOSE) up -d
	@echo ""
	@echo "  Services:"
	@echo "    Airflow UI   → http://localhost:8080  (admin / admin)"
	@echo "    Spark UI     → http://localhost:8081"
	@echo "    API          → http://localhost:8000/docs"
	@echo "    Grafana      → http://localhost:3000  (admin / admin)"
	@echo "    Prometheus   → http://localhost:9090"
	@echo ""

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f

build:
	$(COMPOSE) build --no-cache

# ─── Database ─────────────────────────────────────────────────────────────────
init-db:
	$(COMPOSE) exec $(PG_CTR) psql -U postgres -d clinical_rag \
	  -f /docker-entrypoint-initdb.d/01_init.sql

verify-mimic:
	$(COMPOSE) exec airflow-scheduler python /opt/airflow/dags/../data/scripts/verify_mimic.py

# ─── Spark jobs (manual trigger) ──────────────────────────────────────────────
run-spark-bronze:
	$(COMPOSE) exec spark-master /opt/spark/bin/spark-submit \
	  --master spark://spark-master:7077 \
	  --packages io.delta:delta-core_2.12:2.4.0 \
	  --conf spark.driver.extraJavaOptions=-Dpython.path=/opt/spark-apps \
	  --py-files /opt/spark-apps/spark/utils/spark_session.py \
	  /opt/spark-apps/jobs/bronze_ingest.py

run-spark-silver:
	$(COMPOSE) exec spark-master /opt/spark/bin/spark-submit \
	  --master spark://spark-master:7077 \
	  --packages io.delta:delta-core_2.12:2.4.0 \
	  /opt/spark-apps/jobs/silver_transform.py

run-spark-gold:
	$(COMPOSE) exec spark-master /opt/spark/bin/spark-submit \
	  --master spark://spark-master:7077 \
	  --packages io.delta:delta-core_2.12:2.4.0 \
	  /opt/spark-apps/jobs/gold_sofa_score.py

embed-notes:
	$(COMPOSE) exec $(API_CTR) python -m embeddings.pgvector_loader --backfill

# ─── Dev tooling ──────────────────────────────────────────────────────────────
shell-api:
	$(COMPOSE) exec $(API_CTR) /bin/bash

shell-postgres:
	$(COMPOSE) exec $(PG_CTR) psql -U postgres -d clinical_rag

test:
	$(COMPOSE) exec $(API_CTR) pytest tests/ -v --tb=short

lint:
	ruff check .

format:
	ruff format .
