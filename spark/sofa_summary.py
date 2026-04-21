from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("sofa_summary") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load("/opt/data/gold/sofa_scores")
print("\n=== SOFA Score Distribution ===")
df.groupBy("risk_tier").agg(
    F.count("*").alias("count"),
    F.avg("sofa_total").alias("avg_sofa"),
    F.max("sofa_total").alias("max_sofa")
).orderBy("avg_sofa", ascending=False).show()

print("\n=== Top 10 Highest Risk Patients ===")
df.orderBy("sofa_total", ascending=False).show(10)

print("\nTotal records:", df.count())
spark.stop()
