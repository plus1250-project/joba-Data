import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.functions import col, lit, count
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number

from datetime import datetime, timedelta
from dateutil.tz import gettz
from dateutil.relativedelta import relativedelta


newsSchema = StructType([
    StructField("article_title", StringType(), False),
    StructField("url", StringType(), False),
    StructField("article_text", StringType(), False),
    StructField("press", StringType(), False),
    StructField("issue_date", DateType(), False),
    StructField("industry_name", StringType(), False),
    StructField("keyword", StringType(), True)])


# 산업군 기사별 키워드 리스트 불러오기
keywordDf = spark.read.format("csv").schema(newsSchema).option("dateFormat", "yyyy.MM.dd").load("/data/model/keyword_extraction/*.csv")


# 산업군 키워드별 집계 처리 & 집계일자 컬럼 추가
fromDate = (datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y-%m-%d")
endDate = (datetime.now(gettz('Asia/Seoul'))-relativedelta(months=1)).strftime("%Y-%m-%d")
keywordList = keywordDf.where(col("issue_date").between(endDate, fromDate)).groupBy("industry_name","keyword").count().withColumn("from_date", lit(fromDate))

# 산업군별 키워드 count 순위
widowSpec = Window.partitionBy("industry_name").orderBy(desc("count"))
keywordList = keywordList.withColumn("trend_rank", row_number().over(widowSpec)).where(col("trend_rank") <= 20)

# 컬럼 정렬
keywordList = keywordList.select("industry_name", "from_date", "keyword", col("count").alias("keyword_count"), "trend_rank")


# DB 연결
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "trend_keyword_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "password"}

# DB 저장
keywordList.write.mode("append").jdbc(newPath, tablename, properties=props)

spark.stop()
