"""
Unit tests for SOFA sub-score computation logic.

These tests don't need Spark or a database — we test the scoring functions
in isolation by calling them with PySpark Column literals on a small DataFrame.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("test_sofa")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def _score_df(spark, col_name: str, values: list, score_fn):
    """Helper: create a single-column DF, apply score_fn, return list of scores."""
    df = spark.createDataFrame([(v,) for v in values], [col_name])
    return [row[0] for row in df.select(score_fn(F.col(col_name))).collect()]


class TestRespiratoryScore:
    def test_no_impairment(self, spark):
        from spark.jobs.gold_sofa_score import respiratory_score
        scores = _score_df(spark, "pao2_fio2", [500.0, 400.0], respiratory_score)
        assert scores == [0, 0]

    def test_mild(self, spark):
        from spark.jobs.gold_sofa_score import respiratory_score
        scores = _score_df(spark, "pao2_fio2", [350.0], respiratory_score)
        assert scores == [1]

    def test_severe(self, spark):
        from spark.jobs.gold_sofa_score import respiratory_score
        scores = _score_df(spark, "pao2_fio2", [80.0], respiratory_score)
        assert scores == [4]


class TestCoagulationScore:
    def test_normal(self, spark):
        from spark.jobs.gold_sofa_score import coagulation_score
        scores = _score_df(spark, "platelet", [200.0, 150.0], coagulation_score)
        assert scores == [0, 0]

    def test_critical(self, spark):
        from spark.jobs.gold_sofa_score import coagulation_score
        scores = _score_df(spark, "platelet", [15.0], coagulation_score)
        assert scores == [4]


class TestLiverScore:
    @pytest.mark.parametrize("bilirubin,expected", [
        (0.8, 0),
        (1.5, 1),
        (3.0, 2),
        (8.0, 3),
        (15.0, 4),
    ])
    def test_bilirubin_ranges(self, spark, bilirubin, expected):
        from spark.jobs.gold_sofa_score import liver_score
        scores = _score_df(spark, "bili", [bilirubin], liver_score)
        assert scores[0] == expected


class TestRenalScore:
    @pytest.mark.parametrize("creatinine,expected", [
        (0.9, 0),
        (1.5, 1),
        (2.8, 2),
        (4.2, 3),
        (6.0, 4),
    ])
    def test_creatinine_ranges(self, spark, creatinine, expected):
        from spark.jobs.gold_sofa_score import renal_score
        scores = _score_df(spark, "creat", [creatinine], renal_score)
        assert scores[0] == expected


class TestCNSScore:
    @pytest.mark.parametrize("gcs,expected", [
        (15, 0),
        (13, 1),
        (10, 2),
        (7, 3),
        (4, 4),
    ])
    def test_gcs_ranges(self, spark, gcs, expected):
        from spark.jobs.gold_sofa_score import cns_score
        scores = _score_df(spark, "gcs", [float(gcs)], cns_score)
        assert scores[0] == expected


class TestRiskTier:
    @pytest.mark.parametrize("sofa,expected_tier", [
        (0, "low"),
        (5, "low"),
        (6, "moderate"),
        (9, "moderate"),
        (10, "high"),
        (12, "high"),
        (13, "critical"),
        (22, "critical"),
    ])
    def test_tier_boundaries(self, spark, sofa, expected_tier):
        from spark.jobs.gold_sofa_score import risk_tier
        df = spark.createDataFrame([(float(sofa),)], ["sofa"])
        result = df.select(risk_tier(F.col("sofa"))).collect()[0][0]
        assert result == expected_tier
