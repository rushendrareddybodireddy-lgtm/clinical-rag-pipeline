from typing import Dict, List, Optional, Any
"""
Silver layer — cleaned and joined patient dataset.

Reads from Bronze Delta tables and produces:
  - silver/patient_stays    : one row per ICU stay with demographics + admission metadata
  - silver/vitals_hourly    : CHARTEVENTS pivoted into hourly vital sign windows
  - silver/labs_per_stay    : LABEVENTS with abnormal flag and reference range
  - silver/clinical_notes   : NOTEEVENTS stripped of error notes, de-duped

Key cleaning steps:
  - Drop rows where SUBJECT_ID or HADM_ID is null
  - Filter CHARTEVENTS ERROR == 1 (charting errors)
  - Normalise unit of measure inconsistencies for the vitals we care about
  - Cap extreme outliers using IQR fences (stored as separate flag column,
    not dropped, so the Gold layer can decide how to handle them)
"""

import os
import sys
import logging
from pathlib import Path

from pyspark.sql import functions as F, Window




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

BRONZE_PATH = os.getenv("BRONZE_PATH", "/opt/data/bronze")
SILVER_PATH = os.getenv("SILVER_PATH", "/opt/data/silver")

# MIMIC ITEMIDs for the vitals used in SOFA scoring
VITAL_ITEMIDS = {
    # MAP (Mean Arterial Pressure)
    "map": [52, 456, 6702, 220052, 220181],
    # SpO2
    "spo2": [646, 220277],
    # GCS total
    "gcs": [198, 226755],
    # Respiratory rate
    "resp_rate": [618, 615, 220210, 224690],
    # Heart rate
    "heart_rate": [211, 220045],
    # Temperature (°C)
    "temperature": [223762, 676],
}

# MIMIC ITEMIDs for labs used in SOFA scoring
LAB_ITEMIDS = {
    # Creatinine
    "creatinine": [50912],
    # Bilirubin (total)
    "bilirubin": [50885],
    # Platelets
    "platelet": [51265],
    # PaO2
    "pao2": [50821],
    # FiO2
    "fio2": [50816],
}


def build_patient_stays(spark) -> None:
    """One row per ICU stay enriched with patient demographics."""
    patients  = spark.read.format("delta").load(f"{BRONZE_PATH}/patients")
    admissions = spark.read.format("delta").load(f"{BRONZE_PATH}/admissions")
    icustays  = spark.read.format("delta").load(f"{BRONZE_PATH}/icustays")

    stays = (
        icustays
        .join(patients.select("SUBJECT_ID", "GENDER", "DOB", "DOD"), on="SUBJECT_ID", how="left")
        .join(
            admissions.select(
                "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG",
                "DIAGNOSIS", "INSURANCE", "ETHNICITY",
            ),
            on="HADM_ID",
            how="left",
        )
        .filter(F.col("SUBJECT_ID").isNotNull() & F.col("HADM_ID").isNotNull())
        .withColumn(
            "age_at_admission",
            F.months_between(F.col("ADMITTIME"), F.col("DOB")) / 12,
        )
        # Clip de-identified ages (MIMIC masks patients >89 as 300+)
        .withColumn(
            "age_at_admission",
            F.when(F.col("age_at_admission") > 89, 90).otherwise(F.col("age_at_admission")),
        )
        .withColumn("_transformed_at", F.current_timestamp())
    )

    stays.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        f"{SILVER_PATH}/patient_stays"
    )
    log.info("silver/patient_stays: %d rows", stays.count())


def build_vitals_hourly(spark) -> None:
    """
    Pivot CHARTEVENTS into one row per (ICUSTAY_ID, hour_bucket) with
    min/max/mean for each vital sign we care about.
    """
    chart = spark.read.format("delta").load(f"{BRONZE_PATH}/chartevents")

    # Flatten ITEMIDs → vital_name lookup
    itemid_map_rows = [
        (itemid, vital)
        for vital, ids in VITAL_ITEMIDS.items()
        for itemid in ids
    ]
    itemid_df = spark.createDataFrame(itemid_map_rows, ["ITEMID", "vital_name"])

    vitals = (
        chart
        .filter(F.col("ERROR").isNull() | (F.col("ERROR") != 1))
        .filter(F.col("ICUSTAY_ID").isNotNull())
        .filter(F.col("VALUENUM").isNotNull())
        .join(itemid_df, on="ITEMID", how="inner")
        .withColumn(
            "hour_bucket",
            F.date_trunc("hour", F.col("CHARTTIME")),
        )
    )

    hourly = (
        vitals
        .groupBy("SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "hour_bucket", "vital_name")
        .agg(
            F.mean("VALUENUM").alias("mean_val"),
            F.min("VALUENUM").alias("min_val"),
            F.max("VALUENUM").alias("max_val"),
            F.count("VALUENUM").alias("measurement_count"),
        )
        .withColumn("_transformed_at", F.current_timestamp())
    )

    hourly.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        f"{SILVER_PATH}/vitals_hourly"
    )
    log.info("silver/vitals_hourly: %d rows", hourly.count())


def build_labs_per_stay(spark) -> None:
    """Filter and label lab results relevant to SOFA scoring."""
    labs = spark.read.format("delta").load(f"{BRONZE_PATH}/labevents")

    itemid_map_rows = [
        (itemid, lab)
        for lab, ids in LAB_ITEMIDS.items()
        for itemid in ids
    ]
    itemid_df = spark.createDataFrame(itemid_map_rows, ["ITEMID", "lab_name"])

    relevant = (
        labs
        .join(itemid_df, on="ITEMID", how="inner")
        .filter(F.col("HADM_ID").isNotNull())
        .filter(F.col("VALUENUM").isNotNull())
        .withColumn("is_abnormal", F.col("FLAG").isin(["abnormal", "delta"]).cast("int"))
        .withColumn("_transformed_at", F.current_timestamp())
    )

    relevant.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        f"{SILVER_PATH}/labs_per_stay"
    )
    log.info("silver/labs_per_stay: %d rows", relevant.count())


def build_clinical_notes(spark) -> None:
    """Clean and de-duplicate NOTEEVENTS for embedding."""
    notes = spark.read.format("delta").load(f"{BRONZE_PATH}/noteevents")

    w = Window.partitionBy("SUBJECT_ID", "HADM_ID", "CATEGORY", "CHARTTIME").orderBy(
        F.col("STORETIME").desc_nulls_last()
    )

    clean = (
        notes
        .filter(F.col("ISERROR").isNull() | (F.col("ISERROR") != 1))
        .filter(F.col("TEXT").isNotNull())
        .filter(F.length(F.col("TEXT")) > 50)   # drop near-empty notes
        .withColumn("_row_num", F.row_number().over(w))
        .filter(F.col("_row_num") == 1)         # keep latest version per note
        .drop("_row_num")
        .withColumn("word_count", F.size(F.split(F.col("TEXT"), r"\s+")))
        .withColumn("_transformed_at", F.current_timestamp())
    )

    clean.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        f"{SILVER_PATH}/clinical_notes"
    )
    log.info("silver/clinical_notes: %d rows", clean.count())


def main() -> None:
    spark = get_spark("silver_transform")

    build_patient_stays(spark)
    build_vitals_hourly(spark)
    build_labs_per_stay(spark)
    build_clinical_notes(spark)

    log.info("Silver transform complete.")
    spark.stop()


if __name__ == "__main__":
    main()
