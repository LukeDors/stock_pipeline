import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, to_date, when, trim, regexp_replace
from pyspark.sql.types import DoubleType, DateType

#job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_BUCKET', 'OUTPUT_BUCKET'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

#s3 paths
input_path = f"s3://{args['INPUT_BUCKET']}/raw/"
output_path = f"s3://{args['OUTPUT_BUCKET']}/cleaned/"

print(f"Reading data from: {input_path}")

#read and clean data
df = spark.read.parquet(input_path)

print(f"Initial record count: {df.count()}")
print("Schema before cleaning:")
df.printSchema()

#remove duplucates
df = df.dropDuplicates()

#remove ticker column
distinct_tickers = df.select("Ticker").distinct()
w = Window.orderBy("Ticker")
mapping = distinct_tickers.withColumn("ticker_id", F.row_number().over(w))
df = df.join(mapping, on="Ticker", how="left")
df = df.drop("Ticker")

#handle missing values
critical_columns = ['Date', 'Close', 'Volume']
for column in critical_columns:
    if column in df.columns:
        df = df.filter(col(column).isNotNull())

#handle date col
if 'Date' in df.columns:
    df = df.withColumn('Date', to_date(col('Date')))
    df = df.filter(col('Date').isNotNull())

#clean numeric columns
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Dividends', 'Volume', 'Stock Splits']
for column in numeric_columns:
    if column in df.columns:
        df = df.withColumn(
            column,
            regexp_replace(col(column).cast('string'), '[^0-9.]', '').cast(DoubleType())
        )

if 'Date' in df.columns:
    df = df.orderBy('Date')

# 8. Calculate daily returns if Close price exists
if 'Close' in df.columns:
    from pyspark.sql.window import Window
    from pyspark.sql.functions import lag
    
    window_spec = Window.orderBy('Date')
    df = df.withColumn('Prev_Close', lag('Close', 1).over(window_spec))
    df = df.withColumn(
        'Daily_Return',
        when(col('Prev_Close').isNotNull(),
             ((col('Close') - col('Prev_Close')) / col('Prev_Close')) * 100)
        .otherwise(None)
    )
    df = df.drop('Prev_Close')

print(f"Final record count: {df.count()}")
print("Schema:")
df.printSchema()

print("Data sample:")
df.show(10, truncate=False)

# Write cleaned data to S3
print(f"Writing cleaned data to: {output_path}")
df.write.mode('overwrite').parquet(output_path)

print("Data cleaning completed successfully!")

job.commit()