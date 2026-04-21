from typing import Dict, List, Optional, Any
"""
Gold layer — SOFA score computation and sepsis risk classification.

The Sequential Organ Failure Assessment (SOFA) score evaluates six organ
systems, each scored 0–4.  A total score ≥ 2 with a suspected infection
meets the Sepsis-3 definition.

Organ sub-scores computed here:

  Respiratory  : PaO2/FiO2 ratio (Horowitz index)
  Coagulation  : Platelet count (×10³/μL)
  Liver        : Total bilirubin (mg/dL)
  Cardiovascular: MAP (mmHg) — vasopressor requirements not available in MIMIC CSV
  CNS (Neuro)  : GCS total score
  Renal        : Creatinine (mg/dL)

Output table  : gold/sofa_scores
  - One row per (ICUSTAY_ID, score_window_end)
  - score_window is the 24h look-back from score_window_end
  - Includes individual sub-scores, total SOFA, and a risk_tier label
"""

import os
import sys
import logging
from pathlib import Path

from pyspark.sql import functions as F, Window
from pyspark.sql.types import IntegerType




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

SILVER_PATH = os.getenv("SILVER_PATH", "/opt/data/silver")
GOLD_PATH   = os.getenv("GOLD_PATH",   "/opt/data/gold")

LOOK_BACK_HOURS = 24   # SOFA uses worst value in preceding 24 h


# ─── Sub-score helpers ────────────────────────────────────────────────────────

def respiratory_score(pao2_fio2: "Column") -> "Column":  # noqa: F821
    """
    0: ≥400  |  1: 300–399  |  2: 200–299  |  3: 100–199  |  4: <100
    """
    return (
        F.when(pao2_fio2 >= 400, 0)
        .when(pao2_fio2 >= 300, 1)
        .when(pao2_fio2 >= 200, 2)
        .when(pao2_fio2 >= 100, 3)
        .otherwise(4)
        .cast(IntegerType())
    )


def coagulation_score(platelets: "Column") -> "Column":
    """
    0: ≥150  |  1: 100–149  |  2: 50–99  |  3: 20–49  |  4: <20
    """
    return (
        F.when(platelets >= 150, 0)
        .when(platelets >= 100, 1)
        .when(platelets >= 50, 2)
        .when(platelets >= 20, 3)
        .otherwise(4)
        .cast(IntegerType())
    )


def liver_score(bilirubin: "Column") -> "Column":
    """
    0: <1.2  |  1: 1.2–1.9  |  2: 2.0–5.9  |  3: 6.0–11.9  |  4: ≥12.0
    """
    return (
        F.when(bilirubin < 1.2, 0)
        .when(bilirubin < 2.0, 1)
        .when(bilirubin < 6.0, 2)
        .when(bilirubin < 12.0, 3)
        .otherwise(4)
        .cast(IntegerType())
    )


def cardiovascular_score(map_val: "Column") -> "Column":
    """
    Simplified (no vasopressor data in MIMIC CSV layer):
    0: MAP ≥70  |  1: MAP <70  |  higher scores require vasopressor info
    """
    return (
        F.when(map_val >= 70, 0)
        .otherwise(1)
        .cast(IntegerType())
    )


def cns_score(gcs: "Column") -> "Column":
    """
    0: GCS 15  |  1: 13–14  |  2: 10–12  |  3: 6–9  |  4: <6
    """
    return (
        F.when(gcs == 15, 0)
        .when(gcs >= 13, 1)
        .when(gcs >= 10, 2)
        .when(gcs >= 6, 3)
        .otherwise(4)
        .cast(IntegerType())
    )


def renal_score(creatinine: "Column") -> "Column":
    """
    0: <1.2  |  1: 1.2–1.9  |  2: 2.0–3.4  |  3: 3.5–4.9  |  4: ≥5.0
    """
    return (
        F.when(creatinine < 1.2, 0)
        .when(creatinine < 2.0, 1)
        .when(creatinine < 3.5, 2)
        .when(creatinine < 5.0, 3)
        .otherwise(4)
        .cast(IntegerType())
    )


def risk_tier(sofa_total: "Column") -> "Column":
    """
    Clinical risk tiers based on SOFA total:
      low      : 0–5   (predicted mortality ~10%)
      moderate : 6–9   (~20–40%)
      high     : 10–12 (~40–60%)
      critical : ≥13   (>80%)
    """
    return (
        F.when(sofa_total <= 5, "low")
        .when(sofa_total <= 9, "moderate")
        .when(sofa_total <= 12, "high")
        .otherwise("critical")
    )


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_sofa(spark) -> None:
    vitals = spark.read.format("delta").load(f"{SILVER_PATH}/vitals_hourly")
    labs   = spark.read.format("delta").load(f"{SILVER_PATH}/labs_per_stay")
    stays  = spark.read.format("delta").load(f"{SILVER_PATH}/patient_stays")

    # Build hourly score windows — score at each hour for every stay
    # We use the worst (most abnormal) value in the preceding 24h window.
    # Strategy: join on ICUSTAY_ID, filter CHARTTIME within [window_start, window_end]

    # Step 1: Generate candidate score timestamps (every hour of each ICU stay)
    hour_seq = (
        stays.select("ICUSTAY_ID", "INTIME", "OUTTIME")
        .withColumn("INTIME", F.col("INTIME").cast("timestamp"))
        .withColumn("OUTTIME", F.col("OUTTIME").cast("timestamp"))
        .withColumn(
            "hours_in_icu",
            (F.unix_timestamp("OUTTIME") - F.unix_timestamp("INTIME")) / 3600,
        )
        # Only score stays longer than 24h to have meaningful look-back
        .filter(F.col("hours_in_icu") >= 24)
        .withColumn(
            "score_window_end",
            F.explode(
                F.sequence(
                    F.col("INTIME") + F.expr("INTERVAL 24 HOURS"),
                    F.col("OUTTIME"),
                    F.expr("INTERVAL 1 HOUR"),
                )
            ),
        )
        .withColumn("score_window_start", F.col("score_window_end") - F.expr("INTERVAL 24 HOURS"))
    )

    # Step 2: Pull worst vitals per window
    vitals_pivot = vitals.withColumnRenamed("hour_bucket", "v_time")

    def worst_vital(vital_name: str, agg_fn=F.min):
        # For MAP, GCS, SpO2: worst = minimum; for resp_rate, heart_rate: clinical context varies
        return (
            hour_seq.join(
                vitals_pivot.filter(F.col("vital_name") == vital_name)
                .select("ICUSTAY_ID", "v_time", "mean_val"),
                on="ICUSTAY_ID",
                how="left",
            )
            .filter(
                (F.col("v_time") >= F.col("score_window_start"))
                & (F.col("v_time") < F.col("score_window_end"))
            )
            .groupBy("ICUSTAY_ID", "score_window_end")
            .agg(agg_fn("mean_val").alias(vital_name))
        )

    map_df   = worst_vital("map",  F.min)
    gcs_df   = worst_vital("gcs",  F.min)
    spo2_df  = worst_vital("spo2", F.min)

    # Step 3: Pull worst labs per window
    labs_wide = labs.withColumnRenamed("CHARTTIME", "l_time")

    def worst_lab(lab_name: str, agg_fn=F.max):
        return (
            hour_seq.join(
                labs_wide.filter(F.col("lab_name") == lab_name)
                .select("HADM_ID", "l_time", "VALUENUM"),
                on=["HADM_ID"] if "HADM_ID" in hour_seq.columns else [],
                how="left",
            )
            .filter(
                (F.col("l_time") >= F.col("score_window_start"))
                & (F.col("l_time") < F.col("score_window_end"))
            )
            .groupBy("ICUSTAY_ID", "score_window_end")
            .agg(agg_fn("VALUENUM").alias(lab_name))
        )

    # Re-join hour_seq with HADM_ID for lab joins
    hour_seq_h = hour_seq.join(
        stays.select("ICUSTAY_ID", "HADM_ID"), on="ICUSTAY_ID", how="left"
    )

    def worst_lab_h(lab_name: str, agg_fn=F.max):
        return (
            hour_seq_h.join(
                labs_wide.filter(F.col("lab_name") == lab_name)
                .select("HADM_ID", "l_time", "VALUENUM"),
                on="HADM_ID",
                how="left",
            )
            .filter(
                (F.col("l_time") >= F.col("score_window_start"))
                & (F.col("l_time") < F.col("score_window_end"))
            )
            .groupBy("ICUSTAY_ID", "score_window_end")
            .agg(agg_fn("VALUENUM").alias(lab_name))
        )

    creatinine_df  = worst_lab_h("creatinine",  F.max)
    bilirubin_df   = worst_lab_h("bilirubin",   F.max)
    platelet_df    = worst_lab_h("platelet",    F.min)
    pao2_df        = worst_lab_h("pao2",        F.min)
    fio2_df        = worst_lab_h("fio2",        F.min)

    # Step 4: Assemble all features onto the hour_seq spine
    base = hour_seq.select("ICUSTAY_ID", "score_window_end", "score_window_start")

    for df, key in [
        (map_df,        "map"),
        (gcs_df,        "gcs"),
        (spo2_df,       "spo2"),
        (creatinine_df, "creatinine"),
        (bilirubin_df,  "bilirubin"),
        (platelet_df,   "platelet"),
        (pao2_df,       "pao2"),
        (fio2_df,       "fio2"),
    ]:
        base = base.join(df, on=["ICUSTAY_ID", "score_window_end"], how="left")

    # Step 5: Compute PaO2/FiO2 ratio and SOFA sub-scores
    scored = (
        base
        .withColumn(
            "pao2_fio2",
            F.when(
                F.col("fio2").isNotNull() & (F.col("fio2") > 0),
                F.col("pao2") / (F.col("fio2") / 100.0),
            ).otherwise(None),
        )
        .withColumn("sofa_resp",   respiratory_score(F.col("pao2_fio2")))
        .withColumn("sofa_coag",   coagulation_score(F.col("platelet")))
        .withColumn("sofa_liver",  liver_score(F.col("bilirubin")))
        .withColumn("sofa_cardio", cardiovascular_score(F.col("map")))
        .withColumn("sofa_cns",    cns_score(F.col("gcs")))
        .withColumn("sofa_renal",  renal_score(F.col("creatinine")))
        .withColumn(
            "sofa_total",
            F.col("sofa_resp") + F.col("sofa_coag") + F.col("sofa_liver")
            + F.col("sofa_cardio") + F.col("sofa_cns") + F.col("sofa_renal"),
        )
        .withColumn("risk_tier", risk_tier(F.col("sofa_total")))
        # Sepsis-3: SOFA ≥ 2 increase from baseline — approximate here as total ≥ 2
        .withColumn("sepsis_alert", (F.col("sofa_total") >= 2).cast(IntegerType()))
        .withColumn("_scored_at", F.current_timestamp())
    )

    scored.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        f"{GOLD_PATH}/sofa_scores"
    )
    log.info("gold/sofa_scores: %d rows", scored.count())


def main() -> None:
    spark = get_spark("gold_sofa_score")
    compute_sofa(spark)
    log.info("Gold SOFA scoring complete.")
    spark.stop()


if __name__ == "__main__":
    main()
