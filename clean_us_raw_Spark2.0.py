# Note, this is using Spark 2.0 on Spark Python Notebook, some features are unique here

# cell 1
%sql
-- import US_raw.txt into Spark Cloud first

CREATE TEMPORARY TABLE US_Code
USING com.databricks.spark.csv
OPTIONS (path "[your HDFS path]/US_raw.txt", header "true", delimiter "\t")


# cell 2
## This is the new change in Spark 2.0! Much more convenient!!
us_code_df = spark.sql("select distinct Name as US_City, SubDiv as US_STATE from US_Code")
us_code_df.show(n = 3)
