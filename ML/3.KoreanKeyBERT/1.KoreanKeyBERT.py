#!pip install sentence_transformers
#!pip install konlpy
#!pip install sklearn
#!pip show scikit-learn

import numpy as np
import itertools
import pandas as pd
import dataframe_image as dfi
import torch, os
from datetime import datetime
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

#cleansing 파일 불러오기
dir_path = '/data/collect/article_cleansing'
path = '{}/{}.csv'.format(dir_path,(datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y%m%d"))

df = pd.read_csv(path, header=None)
df0 = df[df[5]=='금융업'].reset_index(drop=True).sample(n=100)
df1 = df[df[5]=='IT정보통신업'].reset_index(drop=True).sample(n=100)
df2 = df[df[5]=='건설업'].reset_index(drop=True).sample(n=100)
df3 = df[df[5]=='화학제약'].reset_index(drop=True).sample(n=100)
df4 = df[df[5]=='음식료업'].reset_index(drop=True).sample(n=100)
df5 = df[df[5]=='기계장비'].reset_index(drop=True).sample(n=100)
df6 = df[df[5]=='판매유통'].reset_index(drop=True).sample(n=100)

df_final = pd.concat([df0,df1,df2,df3,df4,df5,df6]).reset_index(drop=True)

news = []
for i in df_final[2]:
    news.append(i)

keywords_list = []
cnt= 0
for line in news:
    doc = line
    okt = Okt()
    cnt += 1
    stop_words = "국내 코리아 기사 뉴스 오전 오후 매출 상위 하위 상승 하락 증가 감소 대한민국 한국 서울 서울시 부산 부산시 대구 대구시 인천시 인천 광주시 광주 대전시 대전 울산 울산시 경기도 강원도 충북 충청북도 충남 충청남도 전라북도 전북 전라남도 전남 경상북도 경북 경상남도 경남 제주 제주도 세종시 강남구 강동구 강북구 강서구 관악구 광진구 구로구 금천구 노원구 도봉구 동대문구 동작구 마포구 서대문구 서초구 성동구 성북구 송파구 양천구 영등포구 용산구 은평구 종로구 중구 중랑구 중구 서구 동구 영도구 부산진구 동래구 남구 북구 해운대구 사하구 금정구 강서구 연제구 수영구 사상구 기장군 동구 서구 수성구 달서구 달성군 미추홀구 연수구 남동구 부평구 계양구 강화군 옹진군 광산구 유성구 대덕구 울주군 수원시 고양시 용인시 성남시 부천시 화성시 안산시 남양주시 안양시 평택시 시흥시 파주시 의정부시 김포시 광주시 광명시 군포시 하남시 오산시 양주시 이천시 구리시 안성시 포천시 의왕시 양평군 여주시 동두천시 가평군 과천시 연천군 춘천시 원주시 강릉시 동해시 태백시 속초시 삼척시 홍천군 횡성군 영월군 평창군 정선군 철원군 화천군 양구군 인제군 고성군 양양군 청주시 충주시 제천시 보은군 옥천군 영동군 증평군 진천군 괴산군 음성군 단양군 천안시 공주시 보령시 아산시 서산시 논산시 계룡시 당진시 금산군 부여군 서천군 청양군 홍성군 예산군 태안군 전주시 군산시 익산시 정읍시 남원시 김제시 완주군 진안군 무주군 장수군 임실군 순창군 고창군 부안군 목포시 여수시 순천시 나주시 광양시 담양군 곡성군 구례군 고흥군 보성군 화순군 장흥군 강진군 해남군 영암군 무안군 함평군 영광군 장성군 완도군 진도군 신안군 포항시 경주시 김천시 안동시 구미시 영주시 영천시 상주시 문경시 경산시 군위군 의성군 청송군 영양군 영덕군 청도군 고령군 성주군 칠곡군 예천군 봉화군 울진군 울릉군 창원시 진주시 통영시 사천시 김해시 밀양시 거제시 양산시 의령군 함안군 창녕군 고성군 남해군 하동군 산청군 함양군 거창군 합천군 제주시 서귀포시 경기도 한국 경향신문 국민일보 동아일보 문화일보 서울신문 세계일보 조선일보 중앙일보 한겨레 한국일보 뉴스1 뉴시스 연합뉴스 연합뉴스TV 채널A 한국경제TV JTBC KBS MBC MBN SBS Biz TV조선 YTN 매일경제 머니투데이 비즈니스워치 서울경제 아시아경제 이데일리 조선비즈 조세일보 파이낸셜뉴스 한국경제 헤럴드경제 노컷뉴스 더팩트 데일리안 머니S 미디어오늘 아이뉴스24 오마이뉴스 프레시안 디지털데일리 디지털타임스 블로터 전자신문 ZDNet 레이디경향 매경이코노미 시사IN 시사저널 신동아 이코노미스트 주간경향 주간동아 주간조선 중앙SUNDAY 한겨레21 한경비즈니스 기자협회보 농민신문 뉴스타파 동아사이언스 여성신문 일다 코리아중앙데일리 코리아헤럴드 코메디닷컴 헬스조선 강원도민일보 강원일보 국제신문 대구MBC 대전일보 매일신문 부산일보 전주MBC CJB청주방송 JIBS kbc광주방송 신화사 AP연합뉴스 EPA연합뉴스"
    stop_words = set(stop_words.split(' '))
    tokenized_doc = okt.pos(doc)
    tokenized_nouns = [word[0] for word in tokenized_doc if word[1] == 'Noun']
    result = ' '.join([word for word in tokenized_nouns if not word in stop_words])
    n_gram_range = (1, 1)
    
    if len(result)==0:
        count = CountVectorizer(ngram_range=n_gram_range).fit(['none'])
    else:
        count = CountVectorizer(ngram_range=n_gram_range).fit([result])

    candidates = count.get_feature_names_out()
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    top_n = 1
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    keywords_list.append(''.join(keywords))

df_final[6] = keywords_list

path1 = "/data/model/keyword_extraction/keyword_extraction_{}.csv".format((datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y%m%d"))
df_final.to_csv(path1, index=False, header=None)
