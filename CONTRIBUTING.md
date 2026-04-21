# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Prerequisites

- Docker Desktop with at least 6GB memory allocated
- Python 3.11+
- MIMIC-III demo dataset from PhysioNet

## Local Development Setup

1. Clone the repo and copy .env.example to .env
2. Add your Anthropic or OpenAI API key
3. Run docker compose up -d
4. Follow the pipeline steps in README

## Running Tests

pytest tests/ -v

## Known Issues

- NOTEEVENTS is empty in the MIMIC-III demo dataset, so the embeddings pipeline and pgvector retrieval are untested with real clinical notes
- Silver transform requires at least 2GB driver memory due to CHARTEVENTS size (758K rows)
- First spark-submit run takes 3-5 minutes to download Delta Lake JARs

## Code Style

- Follow PEP8 for Python files
- Keep Spark jobs self-contained (no shared module imports due to PySpark classpath issues)
- Add logging for every major transformation step

## Submitting Changes

1. Fork the repo
2. Create a feature branch
3. Commit your changes with descriptive messages
4. Push and open a Pull Request
