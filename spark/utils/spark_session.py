"""
Shared SparkSession factory for all pipeline jobs.

Delta Lake packages are expected to be passed via spark-submit --packages.
This module just sets the consistent AppName and Delta catalog config so
every job uses the same session without repeating boilerplate.
"""

from pyspark.sql import SparkSession


def get_spark(app_name: str, master: str = "local[*]") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Keeps small jobs from spinning up too many shuffle partitions
        .config("spark.sql.shuffle.partitions", "8")
        # Needed for Delta MERGE on large tables
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
