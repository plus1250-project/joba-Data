import pandas as pd
import exchange_calendars as ecals
from pykrx import stock

from datetime import datetime, timedelta
from dateutil.tz import gettz


# 월말 날짜
endOfMonth = (datetime.now(gettz('Asia/Seoul'))).strftime("%Y-%m-%d")

# 한국 코드
XKRX = ecals.get_calendar("XKRX")
# 월말 휴장일 체크 True or False
checkHoliday = XKRX.is_session(endOfMonth)

# 휴장일일 경우 -1day
cnt = 0 
while not checkHoliday :
    cnt += 1
    endOfMonth = (datetime.now(gettz('Asia/Seoul'))-timedelta(cnt)).strftime("%Y-%m-%d")
    checkHoliday = XKRX.is_session(holiday)


# 시가 총액 확인 하기
stockList = stock.get_market_cap(endOfMonth)
maketcap = stockList.reset_index().drop(columns=['종가', '거래량', '거래대금', '상장주식수'])
maketcap.columns = ['stock_code','maket_cap']


# 산업군별 기업 리스트 불러오기
corpcode = pd.read_csv("./data/corperation/matching-corpbyinduscode.csv")

# stock_code 앞자리 0 붙이기
corpcode_list = list(corpcode['stock_code'])
a_list = []
for i in corpcode_list:
    a_list.append(format( i , '06'))
corpcode[6] = a_list
corpcode = corpcode.drop(['stock_code'], axis = 1).rename(columns ={ 6 : 'stock_code'})


# join
industryCorpMarketCap = pd.merge(left = corpcode, right =  maketcap, how = "left", on = "stock_code")
# float 출력 포맷 변경
pd.options.display.float_format = '{:.0f}'.format

# 결측치 처리
industryCorpMarketCap = industryCorpMarketCap.dropna()

# 기업 시가총액 리스트
industryCorpMarketCap = industryCorpMarketCap.drop(["corp_code", "induty_code_short", "main_indus_code", "stock_code"], axis = 1)


# 저장
path = "./data/corperation/inudstry_market_cap_{}.csv".format(datetime.now(gettz('Asia/Seoul')).strftime("%Y%m"))
industryCorpMarketCap.to_csv(path,index=False, header=False)