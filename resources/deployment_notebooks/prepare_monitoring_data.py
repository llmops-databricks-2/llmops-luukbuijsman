# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Load and clean synthetic purchase propensity model monitoring data

# COMMAND ----------

import sys
from pathlib import Path

# Locate the project's src directory
src_path = (Path.cwd() / "../../src").resolve()

# Make the package importable
sys.path.insert(0, str(src_path))

print(f"Source directory: {src_path}")
print(f"Directory exists: {src_path.is_dir()}")

from arxiv_curator.config import load_config
from arxiv_curator.monitoring_data import (
    NUMBER_OF_DAYS,
    generate_raw_monitoring_rows,
)


# COMMAND ----------

from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from arxiv_curator.config import load_config
from arxiv_curator.monitoring_data import NUMBER_OF_DAYS, generate_raw_monitoring_rows

dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")
cfg = load_config("../../project_config.yml", env=env)

spark = SparkSession.builder.getOrCreate()
table_name = f"{cfg.full_schema_name}.model_monitoring_daily"

# COMMAND ----------

# Simulate reading a small raw dataset. All fields initially arrive as strings.
raw_records = generate_raw_monitoring_rows()
raw_df = spark.createDataFrame(raw_records)
print(f"Raw input: {raw_df.count()} rows")

# Parse types, reject invalid records, and deduplicate daily snapshots.
processed_df = (
    raw_df.withColumn("snapshot_date", F.expr("try_cast(snapshot_date AS DATE)"))
    .withColumn("num_predictions", F.col("num_predictions").cast("long"))
    .withColumn("feature_null_rate", F.col("feature_null_rate").cast("double"))
    .withColumn("feature_mean", F.col("feature_mean").cast("double"))
    .withColumn("avg_prediction_score", F.col("avg_prediction_score").cast("double"))
    .filter(F.col("snapshot_date").isNotNull())
    .filter(F.col("num_predictions") > 0)
    .filter(F.col("feature_null_rate").between(0.0, 1.0))
    .filter(F.col("avg_prediction_score").between(0.0, 1.0))
    .dropDuplicates(["snapshot_date", "model_name", "feature_name"])
)

row_count = processed_df.count()
assert row_count == NUMBER_OF_DAYS, f"Expected {NUMBER_OF_DAYS} clean rows; got {row_count}"

# An overwrite is deliberate: rerunning the job should not append duplicate days.
processed_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
print(f"Successfully wrote {row_count} rows to {table_name}")

# COMMAND ----------

print("Delta table schema:")
spark.table(table_name).printSchema()
print("Most recent sample records:")
for row in spark.table(table_name).orderBy(F.col("snapshot_date").desc()).limit(5).collect():
    print(row.asDict())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM llmops_dev.luukbuijsman.model_monitoring_daily
# MAGIC ORDER BY snapshot_date DESC;

# COMMAND ----------

