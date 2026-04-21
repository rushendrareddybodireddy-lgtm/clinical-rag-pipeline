"""
DAG: patient_ingestion
Schedule: hourly

Runs the Bronze and Silver Spark jobs sequentially.  In a production setup
you'd use SparkSubmitOperator pointing at your cluster; here we use
BashOperator with spark-submit so it works against the local Spark service
in docker-compose without additional Airflow provider config.

Task graph:
    check_mimic_files → bronze_ingest → silver_transform → notify_complete
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

SPARK_SUBMIT = "spark-submit --master spark://spark-master:7077 --packages io.delta:delta-core_2.12:2.4.0,io.delta:delta-storage:2.4.0"
SPARK_APPS   = "/opt/airflow/spark"

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="patient_ingestion",
    description="Hourly MIMIC-III Bronze → Silver medallion pipeline",
    schedule_interval="@hourly",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["mimic", "bronze", "silver", "medallion"],
) as dag:

    def _check_mimic_files(**ctx):
        """Fail fast if MIMIC CSVs are not mounted."""
        mimic_path = os.getenv("MIMIC_DATA_PATH", "/opt/mimic")
        required = ["PATIENTS.csv", "ADMISSIONS.csv", "ICUSTAYS.csv"]
        missing = [f for f in required if not os.path.exists(f"{mimic_path}/{f}")]
        if missing:
            raise FileNotFoundError(
                f"MIMIC-III CSV files not found at {mimic_path}: {missing}\n"
                f"Mount your MIMIC-III data directory to {mimic_path} in docker-compose.yml"
            )

    check_mimic_files = PythonOperator(
        task_id="check_mimic_files",
        python_callable=_check_mimic_files,
    )

    bronze_ingest = BashOperator(
        task_id="bronze_ingest",
        bash_command=f"{SPARK_SUBMIT} {SPARK_APPS}/jobs/bronze_ingest.py",
        env={
            "MIMIC_DATA_PATH": "{{ var.value.get('mimic_data_path', '/opt/mimic') }}",
            "BRONZE_PATH": "/opt/data/bronze",
        },
    )

    silver_transform = BashOperator(
        task_id="silver_transform",
        bash_command=f"{SPARK_SUBMIT} {SPARK_APPS}/jobs/silver_transform.py",
        env={
            "BRONZE_PATH": "/opt/data/bronze",
            "SILVER_PATH": "/opt/data/silver",
        },
    )

    def _notify_complete(**ctx):
        run_id = ctx["run_id"]
        print(f"[patient_ingestion] Run {run_id} completed successfully — Silver tables ready.")

    notify_complete = PythonOperator(
        task_id="notify_complete",
        python_callable=_notify_complete,
    )

    check_mimic_files >> bronze_ingest >> silver_transform >> notify_complete
