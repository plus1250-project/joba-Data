import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.functions import col
from pyspark.sql.functions import month
from pyspark.sql.functions import desc

from datetime import datetime, timedelta
from dateutil.tz import gettz

newsSchema = StructType([
    StructField("article_title", StringType(), False),
    StructField("url", StringType(), False),
    StructField("article_text", StringType(), True),
    StructField("press", StringType(), True),
    StructField("issue_date", DateType(), False),
    StructField("industry_name", StringType(), True)])

# 전날 기사 파일 불러오기
path = "/data/model/article_classification/article_classification_{}.csv".format((datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y%m%d"))
articleDF = spark.read.format("csv").schema(newsSchema).load(path).na.drop("any")

# 기타 카테고리 제거
articleList = articleDF.select("industry_name", "url", "issue_date", "article_title", "press").where(col("industry_name") != "기타")

# DB 연결
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "industry_article_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "password"}

# DB 저장
articleList.write.mode("append").jdbc(newPath, tablename, properties=props)

spark.stop()




