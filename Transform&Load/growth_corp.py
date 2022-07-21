import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col, expr, lit, round
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number

from datetime import datetime
from dateutil.tz import gettz
from dateutil.relativedelta import relativedelta

newsSchema = StructType([
    StructField("corp_name", StringType(), False),
    StructField("industry_name", StringType(), False),
    StructField("maket_cap", FloatType(), False)])


# 당월 산업군별 기업 시가총액 불러오기
path = "/data/collect/corperation/inudstry_market_cap_{}.csv".format(datetime.now(gettz('Asia/Seoul')).strftime("%Y%m"))
corpDf = spark.read.format("csv").schema(newsSchema).load(path)

# 전월 산업군별 기업 시가총액 불러오기
path1 = "/data/collect/corperation/inudstry_market_cap_{}.csv".format((datetime.now(gettz('Asia/Seoul'))-relativedelta(months=1)).strftime("%Y%m"))
preMonth_corpDf = spark.read.format("csv").schema(newsSchema).load(path1)


# preMonth_corpDf 컬럼명 변경
preMonth_corpDf = preMonth_corpDf.select(col('corp_name').alias('corp_name1') ,col('industry_name').alias('industry_name1'), col('maket_cap').alias('maket_cap1'))

# 당월 & 전월 산업군별 기업 시가총액 left outer join
joinExprssion = (corpDf["corp_name"] == preMonth_corpDf["corp_name1"]) & (corpDf["industry_name"] == preMonth_corpDf["industry_name1"])
growth_corp = corpDf.join(preMonth_corpDf, joinExprssion, "left_outer")

# 전월 대비 시가총액 증가율 처리
growth_corp = growth_corp.select('corp_name', 'industry_name', round(expr('((maket_cap-maket_cap1)/maket_cap1)*100'), 2).alias('growth_rate'))

# 시가총액 증가율 순위 집계
widowSpec = Window.partitionBy("industry_name").orderBy(desc("growth_rate"))
growth_corp = growth_corp.withColumn("corp_rank", row_number().over(widowSpec)).where(col("corp_rank") <= 10)

# 집계 발행월 컬럼 추가
growth_corp = growth_corp.withColumn("reg_month", lit(datetime.now(gettz('Asia/Seoul')).strftime("%Y-%m")))

# 기타 카테고리 제거
growth_corp = growth_corp.select("industry_name", "reg_month", "corp_name","growth_rate", "corp_rank").where(col("industry_name") != "기타")


# DB 연결
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "growth_corp_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "password"}

# DB 저장
growth_corp.write.mode("append").jdbc(newPath, tablename, properties=props)


spark.stop()
