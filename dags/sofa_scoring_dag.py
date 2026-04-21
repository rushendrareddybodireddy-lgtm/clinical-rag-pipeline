"""
DAG: sofa_scoring
Schedule: hourly (offset 20 min so it runs after patient_ingestion finishes)

Runs the Gold SOFA scoring Spark job, then writes the latest high-risk
patient summary into Postgres so the FastAPI endpoint can serve it
without hitting Spark at query time.

Task graph:
    wait_for_silver → compute_sofa_scores → write_alerts_to_postgres
"""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

SPARK_SUBMIT = "spark-submit --master spark://spark-master:7077 --packages io.delta:delta-core_2.12:2.4.0,io.delta:delta-storage:2.4.0"
SPARK_APPS   = "/opt/airflow/spark"

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="sofa_scoring",
    description="Hourly SOFA score computation and sepsis alert generation",
    schedule_interval="20 * * * *",   # :20 past every hour
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["sofa", "gold", "sepsis", "alerts"],
) as dag:

    wait_for_silver = ExternalTaskSensor(
        task_id="wait_for_silver",
        external_dag_id="patient_ingestion",
        external_task_id="notify_complete",
        mode="reschedule",
        poke_interval=120,
        timeout=1800,   # give up after 30 min — don't block next run
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
    )

    compute_sofa_scores = BashOperator(
        task_id="compute_sofa_scores",
        bash_command=f"{SPARK_SUBMIT} {SPARK_APPS}/jobs/gold_sofa_score.py",
        env={
            "SILVER_PATH": "/opt/data/silver",
            "GOLD_PATH": "/opt/data/gold",
        },
    )

    def _write_alerts_to_postgres(**ctx):
        """
        Read the latest SOFA scores from Delta and upsert alert rows into
        the Postgres `sepsis_alerts` table so the API can serve them fast.
        """
        import os
        import psycopg2
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        spark = (
            SparkSession.builder.appName("sofa_alerts_writer")
            .master("local[2]")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .getOrCreate()
        )

        gold_path = os.getenv("GOLD_PATH", "/opt/data/gold")
        silver_path = os.getenv("SILVER_PATH", "/opt/data/silver")

        # Grab latest score per ICU stay
        latest_scores = (
            spark.read.format("delta").load(f"{gold_path}/sofa_scores")
            .filter(F.col("sepsis_alert") == 1)
            .withColumn(
                "_rank",
                F.row_number().over(
                    __import__("pyspark.sql", fromlist=["Window"])
                    .Window.partitionBy("ICUSTAY_ID")
                    .orderBy(F.col("score_window_end").desc())
                ),
            )
            .filter(F.col("_rank") == 1)
            .select(
                "ICUSTAY_ID", "score_window_end", "sofa_total",
                "risk_tier", "sofa_resp", "sofa_coag", "sofa_liver",
                "sofa_cardio", "sofa_cns", "sofa_renal",
            )
        )

        rows = latest_scores.collect()
        spark.stop()

        pg = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DB", "clinical_rag"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )
        cur = pg.cursor()

        upsert_sql = """
            INSERT INTO sepsis_alerts
                (icustay_id, score_window_end, sofa_total, risk_tier,
                 sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_cns, sofa_renal,
                 created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (icustay_id)
            DO UPDATE SET
                score_window_end = EXCLUDED.score_window_end,
                sofa_total       = EXCLUDED.sofa_total,
                risk_tier        = EXCLUDED.risk_tier,
                sofa_resp        = EXCLUDED.sofa_resp,
                sofa_coag        = EXCLUDED.sofa_coag,
                sofa_liver       = EXCLUDED.sofa_liver,
                sofa_cardio      = EXCLUDED.sofa_cardio,
                sofa_cns         = EXCLUDED.sofa_cns,
                sofa_renal       = EXCLUDED.sofa_renal,
                updated_at       = NOW();
        """

        for row in rows:
            cur.execute(upsert_sql, (
                row.ICUSTAY_ID, row.score_window_end, row.sofa_total, row.risk_tier,
                row.sofa_resp, row.sofa_coag, row.sofa_liver,
                row.sofa_cardio, row.sofa_cns, row.sofa_renal,
            ))

        pg.commit()
        cur.close()
        pg.close()

        print(f"[sofa_scoring] Upserted {len(rows)} sepsis alert rows into Postgres.")

    write_alerts_to_postgres = PythonOperator(
        task_id="write_alerts_to_postgres",
        python_callable=_write_alerts_to_postgres,
    )

    wait_for_silver >> compute_sofa_scores >> write_alerts_to_postgres
