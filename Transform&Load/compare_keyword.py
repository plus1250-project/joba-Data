import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import date_format
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number

from datetime import datetime
from dateutil.tz import gettz
from dateutil.relativedelta import relativedelta


# DB 읽기
keywordDF = spark.read.format("jdbc")\
    .option("url","jdbc:mysql://172.31.30.12:3306/joba")\
    .option("driver","com.mysql.cj.jdbc.Driver")\
    .option("dbtable","month_rank_keyword_list")\
    .option("user","user").option("password","password").load()

# 전월 month_rank_keyword_list 불러오기
regMonth = (datetime.now(gettz('Asia/Seoul'))-relativedelta(months=1)).strftime("%Y-%m")
keywordDF = keywordDF.where(col("reg_month") == regMonth)


# 전월 랭킹 키워드 비교 분석
newsSchema = StructType([
    StructField("article_title", StringType(), False),
    StructField("url", StringType(), False),
    StructField("article_text", StringType(), False),
    StructField("press", StringType(), False),
    StructField("issue_date", DateType(), False),
    StructField("industry_name", StringType(), False),
    StructField("keyword", StringType(), True)])


preMonthKeywordDf = spark.read.format("csv").schema(newsSchema).option("dateFormat", "yyyy.MM.dd").load("/data/model/keyword_extraction/*.csv")

# date 포맷 변경 
preMonthKeywordDf = preMonthKeywordDf.select(col("industry_name").alias("industry_name1"), col("keyword").alias("keyword1"), date_format("issue_date", "yyyy-MM").alias("reg_month1"))

# 전전월 keyword count
reg2MonthAgo = (datetime.now(gettz('Asia/Seoul'))-relativedelta(months=2)).strftime("%Y-%m")
preMonthKeywordDf = preMonthKeywordDf.where(col("reg_month1") == reg2MonthAgo).groupBy("industry_name1","keyword1","reg_month1").count()

# 전전월 산업군별 랭킹 순위 집계
widowSpec = Window.partitionBy("industry_name1").orderBy(desc("count"))
preMonthRankKeyword = preMonthKeywordDf.withColumn("count_rank", row_number().over(widowSpec))


#  join
joinExprssion = (keywordDF["industry_name"] == preMonthRankKeyword["industry_name1"]) & (keywordDF["keyword"] == preMonthRankKeyword["keyword1"])
compareKeyword = keywordDF.join(preMonthRankKeyword, joinExprssion, "left_outer")

# 결측치 처리
compareKeyword = compareKeyword.na.fill(0)

# count증가량 & rank 변화량 집계 처리
compareKeyword = compareKeyword.select('industry_name', 'keyword', 'reg_month', expr('keyword_count-count').alias('increment'), expr('count_rank-month_rank').alias('change_rank'), 'month_rank')


# DB 저장
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "compare_keyword_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "password"}

compareKeyword.write.mode("append").jdbc(newPath, tablename, properties=props)


spark.stop()