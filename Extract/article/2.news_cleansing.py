# 불필요 기사 데이터 삭제 처리
import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()
from pyspark.sql.types import StructType, StructField, StringType, DateType
from datetime import datetime, timedelta
from dateutil.tz import gettz

newsSchema = StructType([
    StructField("NEWS_TITLE", StringType(), False),
    StructField("NEWS_URL", StringType(), False),
    StructField("NEWS_TEXT", StringType(), False),
    StructField("NEWS_PRESS", StringType(), False),
    StructField("NEWS_DATE", DateType(), False)])

path = '/data/news/{}_*.csv'.format((datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y%m%d"))
newsDf = spark.read.format("csv").schema(newsSchema).option("dateFormat", "yyyy.MM.dd").load(path)

# 불필요 기사 데이터 확인
newsDf.createOrReplaceTempView("news")
tmp = spark.sql("select * from news where (news_title not like '%[표]%')\
                and (news_title not like '%[인사]%')\
                and (news_title not like '%[부고]%')\
                and (news_title not like '%[부음]%')\
                and (news_title not like '%[화보]%')\
                and (news_title not like '%[독서일기]%')\
                and (news_title not like '%[장외주식]%')\
                and (news_title not like '%[포토뉴스]%')\
                and (news_title not like '%[사진뉴스]%')\
                and (news_title not like '%[오늘의 역사]%')\
                and (news_title not like '%[그립습니다]%')\
                and (news_title not like '%[200자 읽기]%')\
                and (news_title not like '%[공덕포차]%')\
                and (news_title not like '%[임성용의 보약밥상]%')\
                and (news_title not like '%[공독쌤의 공부머리 독서법]%')\
                and (news_title not like '%[박세희·우제원의 독서연애]%')\
                and (news_title not like '%[김민경 ‘맛’ 이야기]%')\
                and (news_title not like '%[에세이]%')\
                and (news_title not like '%[왕초보영어탈출 해커스톡]%')\
                and (news_title not like '%[%날씨%]%')\
                and (news_title not like '%[프로필]%')\
                and (news_title not like '%[영상채록%')\
                and (news_title not like '%[사설]%')\
                and (news_title not like '%만평]%')\
                and (news_title not like '%오늘의 운세%')\
                and (news_title not like '%[개업]%')\
                and (news_title not like '%[행사]%')\
                and (news_title not like '%[%기고%]%')\
                and (news_title not like '%[%칼럼%]%')\
                and (news_title not like '%[2023학년도 논술길잡이]%')\
                and (news_title not like '%[신철수 쌤의 국어 지문 읽기]%')\
                and (news_title not like '%[영어 이야기]%')\
                and (news_title not like '%[홍성호 기자의 열려라! 우리말]%')\
                and (news_title not like '%[두근두근 뇌 운동]%')\
                and (news_title not like '%[지상갤러리]%')\
                and (news_title not like '%[오늘의 주요 일정]%')\
                and (news_title not like '%【동정】%')\
                and (news_title not like '%[동정]%')\
                and (news_title not like '%[뉴스광장 영상]%')\
                and (news_title not like '%[MZ기자 홍성효의 배워봅시다]%')\
                and (news_title not like '%[%마감%]%')\
                and (news_title not like '%[달러/원]%')\
                and (news_title not like '%[스팟]%')\
                and (news_title not like '%[출발]%')\
                and (news_title not like '%[주식 매매 상위 종목 및 환율]%')\
                and (news_title not like '%[주요경제지표]%')\
                and (news_title not like '%[시승기]%')\
                and (news_title not like '%[시론]%')\
                and (news_title not like '%[유럽개장]%')\
                and (news_title not like '%NEWS IN FOCUS%')\
                and (news_title not like '%[취임사]%')\
                and (news_title not like '%[수학 두뇌를 키워라]%')\
                and (news_title not like '%[신동열의 고사성어 읽기]%')\
                and (news_title not like '%[경제·금융 상식 퀴즈 O X]%')\
                and (news_title not like '%[테샛 공부합시다]%')\
                and (news_title not like '%[국가공인 경제이해력 검증시험 맛보기]%')\
                and (news_title not like '%[매경CEO 특강]%')\
                and (news_title not like '%[부고종합]%')\
                and (news_title not like '%[인사종합]%')\
                and (news_title not like '%[설문]%')\
                and (news_title not like '%[Lifehacks]%')\
                and (news_title not like '%[이달의 기자상]%')\
                and (news_title not like '%[뷰파인더 너머]%')\
                and (news_title not like '%[코스닥]%')\
                and (news_title not like '%[코스피]%')\
                and (news_title not like '%[%캘린더%]%')\
                and (news_title not like '%[%시황%]%')\
                and (news_title not like '%[일일펀드동향]%')\
                and (news_title not like '%[주간경제지표]%')\
                and (news_title not like '%[결혼]%')\
                and (news_title not like '%[김병진의 세상보기]%')\
                and (news_title not like '%[부케부캐]%')\
                and (NEWS_PRESS != '코메디닷컴')\
                and (NEWS_PRESS != '헬스조선')")

# ## 기사제목 및 내용 null, 중복 처리
newsNullDropDf = tmp.na.drop()
newsCleansing = newsNullDropDf.dropDuplicates(['NEWS_TITLE', 'NEWS_TEXT'])

## 기사 정제 데이터 저장
path1 = "/data/collect/article_cleansing/article_cleansing_{}.csv".format((datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y%m"))
newsCleansing.write.csv(path1, header = True)