import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col, lit
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number

from datetime import datetime
from dateutil.tz import gettz

newsSchema = StructType([
    StructField("corp_name", StringType(), False),
    StructField("industry_name", StringType(), False),
    StructField("maket_cap", FloatType(), False)])


# 산업군별 기업 시가총액 리스트 불러오기
path = "/data/collect/corperation/inudstry_market_cap_{}.csv".format(datetime.now(gettz('Asia/Seoul')).strftime("%Y%m"))
corpdDf = spark.read.format("csv").schema(newsSchema).load(path)

# 시가총액 순위 집계
widowSpec = Window.partitionBy("industry_name").orderBy(desc("maket_cap"))
corpdDf = corpdDf.withColumn("corp_rank", row_number().over(widowSpec)).where(col("corp_rank") <= 10)

# 집계 발행월 컬럼 추가
corpList = corpdDf.withColumn("reg_month", lit(datetime.now(gettz('Asia/Seoul')).strftime("%Y-%m")))

# 기타 카테고리 제거
industryCorpList = corpList.select("industry_name", "reg_month", "corp_name", "corp_rank").where(col("industry_name") != "기타")

# DB 연결
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "industry_corp_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "password"}

# DB 저장
industryCorpList.write.mode("append").jdbc(newPath, tablename, properties=props)

spark.stop()
