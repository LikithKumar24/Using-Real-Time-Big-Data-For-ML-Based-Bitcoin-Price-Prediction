from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("BTCPriceConsumer") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Define the schema for incoming BTC data
schema = StructType() \
    .add("symbol", StringType()) \
    .add("timestamp", StringType()) \
    .add("open", DoubleType()) \
    .add("close", DoubleType()) \
    .add("high", DoubleType()) \
    .add("low", DoubleType()) \
    .add("volume", DoubleType())

# 3. Read from Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "btc_prices") \
    .option("startingOffsets", "latest") \
    .load()

# 4. Parse JSON data from Kafka
df_parsed = df_raw.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# 5. Output to console (live stream display)
df_parsed.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

# 6. Optional: Save to CSV for ML model training
df_parsed.writeStream \
    .format("csv") \
    .option("path", "C:/btc-data/output") \
    .option("checkpointLocation", "C:/btc-data/checkpoints") \
    .outputMode("append") \
    .start() \
    .awaitTermination()
