from typing import Dict, List, Optional, Any
"""
Bronze layer — raw MIMIC-III CSV ingestion.

Reads the six core MIMIC-III tables from the raw CSV mount and writes each
one as a Delta table under $BRONZE_PATH.  We deliberately keep this layer
un-transformed so the Silver job owns all cleaning logic and this job stays
idempotent (safe to re-run on the same data).

Expected CSV files under $MIMIC_DATA_PATH:
    PATIENTS.csv
    ADMISSIONS.csv
    ICUSTAYS.csv
    CHARTEVENTS.csv      (large — expect 200M+ rows on full MIMIC)
    LABEVENTS.csv        (large)
    NOTEEVENTS.csv
"""

import os
import sys
import logging
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, LongType, StringType, TimestampType,
)

# Allow running as spark-submit from repo root



def get_spark(app_name, master="local[*]"):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName(app_name).master(master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MIMIC_PATH  = os.getenv("MIMIC_DATA_PATH", "/opt/mimic")
BRONZE_PATH = os.getenv("BRONZE_PATH", "/opt/data/bronze")

# ─── Schema hints ─────────────────────────────────────────────────────────────
# We let Spark infer most types but provide explicit casts for the columns we
# know are problematic in the raw CSVs (mixed nulls, scientific notation, etc.)

CAST_MAP: Dict[str, dict] = {
    "PATIENTS": {
        "SUBJECT_ID": IntegerType(),
        "GENDER": StringType(),
        "DOB": TimestampType(),
        "DOD": TimestampType(),
    },
    "ADMISSIONS": {
        "SUBJECT_ID": IntegerType(),
        "HADM_ID": IntegerType(),
        "ADMITTIME": TimestampType(),
        "DISCHTIME": TimestampType(),
        "HOSPITAL_EXPIRE_FLAG": IntegerType(),
    },
    "ICUSTAYS": {
        "SUBJECT_ID": IntegerType(),
        "HADM_ID": IntegerType(),
        "ICUSTAY_ID": IntegerType(),
        "INTIME": TimestampType(),
        "OUTTIME": TimestampType(),
        "LOS": DoubleType(),
    },
    "CHARTEVENTS": {
        "SUBJECT_ID": IntegerType(),
        "HADM_ID": IntegerType(),
        "ICUSTAY_ID": IntegerType(),
        "ITEMID": IntegerType(),
        "CHARTTIME": TimestampType(),
        "VALUENUM": DoubleType(),
        "VALUEUOM": StringType(),
        "ERROR": IntegerType(),
    },
    "LABEVENTS": {
        "SUBJECT_ID": IntegerType(),
        "HADM_ID": IntegerType(),
        "ITEMID": IntegerType(),
        "CHARTTIME": TimestampType(),
        "VALUENUM": DoubleType(),
        "VALUEUOM": StringType(),
        "FLAG": StringType(),
    },
    "NOTEEVENTS": {
        "SUBJECT_ID": IntegerType(),
        "HADM_ID": IntegerType(),
        "CHARTTIME": TimestampType(),
        "CHARTDATE": StringType(),
        "STORETIME": TimestampType(),
        "CATEGORY": StringType(),
        "DESCRIPTION": StringType(),
        "ISERROR": IntegerType(),
        "TEXT": StringType(),
    },
}


def ingest_table(spark, table_name: str) -> None:
    src = f"{MIMIC_PATH}/{table_name}.csv"
    dst = f"{BRONZE_PATH}/{table_name.lower()}"

    if not Path(src).exists():
        log.warning("Skipping %s — file not found at %s", table_name, src)
        return

    log.info("Ingesting %s → %s", table_name, dst)

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")   # always read as string first
        .option("multiLine", "true")      # NOTEEVENTS TEXT can span lines
        .option("escape", '"')
        .csv(src)
    )

    # Apply explicit casts where we know the target type
    for col_name, dtype in CAST_MAP.get(table_name, {}).items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(dtype))

    # Append ingestion metadata
    df = df.withColumn("_ingested_at", F.current_timestamp()).withColumn(
        "_source_file", F.lit(src)
    )

    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(dst)
    )

    count = df.count()
    log.info("  ✓ %s: wrote %d rows → %s", table_name, count, dst)


def main() -> None:
    spark = get_spark("bronze_ingest")
    tables = ["PATIENTS", "ADMISSIONS", "ICUSTAYS", "CHARTEVENTS", "LABEVENTS", "NOTEEVENTS"]

    for t in tables:
        ingest_table(spark, t)

    log.info("Bronze ingestion complete.")
    spark.stop()


if __name__ == "__main__":
    main()
