import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import date_format
from pyspark.sql import functions as F

from datetime import datetime, timedelta
from dateutil.tz import gettz


# DB 읽기
keywordDF = spark.read.format("jdbc")/
    .option("url","jdbc:mysql://172.31.30.12:3306/joba")/
    .option("driver","com.mysql.cj.jdbc.Driver")/
    .option("dbtable","trend_keyword_list")/
    .option("user","user").option("password","password").load()


# 전일 기준 trend_keyword_list 불러오기
fromDate = (datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y-%m-%d")
keywordMonth = keywordDF.where(col("from_date")==fromDate)

# trend_keyword_list 테이블에서 산업군별 키워드 리스트 추출
keywordMonthList = keywordMonth.groupby("industry_name").agg(F.collect_list("keyword"))
indusKeywordList = keywordMonthList.select('collect_list(keyword)').rdd.map(lambda x : x[0]).collect()
industyList = keywordMonthList.select('industry_name').rdd.map(lambda x : x[0]).collect()


# 산업군 기사별 키워드 리스트 불러오기
newsSchema = StructType([
    StructField("article_title", StringType(), False),
    StructField("url", StringType(), False),
    StructField("article_text", StringType(), False),
    StructField("press", StringType(), False),
    StructField("issue_date", DateType(), False),
    StructField("industry_name", StringType(), False),
    StructField("keyword", StringType(), True)])

keywordDf = spark.read.format("csv").schema(newsSchema).option("dateFormat", "yyyy.MM.dd").load("/data/model/keyword_extraction/*.csv")


# DB 옵션값 설정 
newPath = "jdbc:mysql://172.31.30.12:3306/joba"
tablename = "month_keyword_list"
props = {"driver":"com.mysql.cj.jdbc.Driver", "user":"user", "password": "password"}

# DB로부터 추출한 키워드에 대한 월별 집계 처리 & DB 적재
for i in range(7):
    cnt = 0
    for j in indusKeywordList[i]:
        cnt += 1 
        globals()["{}".format(industyList[i])+"{}".format(cnt)] = 
            keywordDf.select("keyword", date_format("issue_date", "yyyy-MM").alias("reg_month"), "industry_name")\
            .where(col("keyword") == j).where(col("industry_name") == industyList[i])\
            .groupby("keyword","reg_month","industry_name").count()\
            .withColumnRenamed("count", "keyword_count")\
            .withColumn("input_date", lit(fromDate)).sort(col("reg_month"))
        
        # keyword_count = 0일 경우 row 추가
        for k in list(set(['2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06']) - set(globals()["{}".format(industyList[i])+"{}".format(cnt)].select('reg_month').rdd.map(lambda x : x[0]).collect())):
            print(k)
            if len(globals()["{}".format(industyList[i])+"{}".format(cnt)].select('reg_month').rdd.map(lambda x : x[0]).collect()) == 12:
                globals()["{}".format(industyList[i])+"{}".format(cnt)]
            else:
                newRow = spark.createDataFrame([(j, '{}'.format(k), industyList[i], 0, fromDate)], ['keyword', 'reg_month', 'industry_name', 'keyword_count', 'input_date'])          
                globals()["{}".format(industyList[i])+"{}".format(cnt)] = globals()["{}".format(industyList[i])+"{}".format(cnt)].union(newRow).sort(col("reg_month"))
                
        globals()["{}".format(industyList[i])+"{}".format(cnt)].show()
        # DB 저장
        globals()["{}".format(industyList[i])+"{}".format(cnt)].write.mode("append").jdbc(newPath, tablename, properties=props)


spark.stop()
