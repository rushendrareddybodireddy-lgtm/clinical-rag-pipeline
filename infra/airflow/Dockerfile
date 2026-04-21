FROM apache/airflow:2.9.1-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres==5.11.2 \
    psycopg2-binary==2.9.9 \
    pyspark==3.5.1 \
    openai==1.30.0 \
    anthropic==0.28.0 \
    pgvector==0.2.5
