import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.functions import col, count
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number

from datetime import datetime
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

# date 포멧 변경
keywordDf= keywordDf.select("industry_name", "keyword", date_format("issue_date", "yyyy-MM").alias("reg_month"))

# 전월 키워드 추출
regMonth = (datetime.now(gettz('Asia/Seoul'))-relativedelta(months=1)).strftime("%Y-%m")
monthKeyword = keywordDf.groupBy("industry_name","keyword","reg_month").count().where(col("reg_month") == regMonth)

# 전월 랭킹 순위 집계
widowSpec = Window.partitionBy("industry_name").orderBy(desc("count"))
monthRankKeyword = monthKeyword.withColumn("month_rank", row_number().over(widowSpec)).where(col("month_rank") <= 10).select("industry_name","reg_month", "keyword", col('count').alias('keyword_count'), "month_rank")


# DB 저장
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "month_rank_keyword_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "passwrod"}

monthRankKeyword.write.mode("append").jdbc(newPath, tablename, properties=props)


spark.stop()

