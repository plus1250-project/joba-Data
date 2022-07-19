import re, os, requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from dateutil.tz import gettz
import pandas as pd

def article(date_, url, head):
    articles = []
    reporter = re.compile("[가-힣]{2,4}\s*기자")
    email   = re.compile("[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,4}")
    f_reporter = re.compile("[가-힣]{2,4}\s*특파원")

    r = requests.get(url, headers= head)
    bs = BeautifulSoup(r.text)
    
    try:
        rt = bs.find("div", id="newsct_article")
        
        # 이미지 캡션 제외
        rt_em = rt.find_all('em','img_desc')
        rt_td = rt.find_all('td')
        for em in rt_em:
            em.decompose()
        for td in rt_td:
            td.decompose()
            
        # 기사내용, 제목, 언론사, 발행날짜 추출    
        text = rt.text.strip().replace(',','').replace('\n','')
        a_title= bs.find("h2")
        title = a_title.text.strip().replace(',','')
        a_press = bs.find("p","c_text")
        press = a_press.text[12:-38]
        a_time = bs.find("span","media_end_head_info_datestamp_time _ARTICLE_DATE_TIME")
        time = a_time.text[:10]   
    except:
        return  
    
    try:
        text = text[:text.find(rt.select_one("a").text)]   
    except:
        pass  
    
    # email, 기자, 특파원 제외
    text = re.sub(email, "", text)
    text = re.sub(reporter, "", text)   
    text = re.sub(f_reporter,"",text)
    
    articles.append(title+','+url+','+ text + ',' +press+','+time + '\n')

    if not os.path.isdir("news"):
        os.mkdir("news")
    f = open("../data/news/" + date_ + "_" + url.split("=")[1] + ".csv", "a", encoding='utf-8')
    f.write('\n'.join(articles))
    f.close()

def naver_news(start_date = None, end_date = None):
    sid1_cate = [100, 101, 102, 103, 104, 105]
    sid2_cate100 = [264, 265, 268, 266, 267, 269]
    sid2_cate101 = [259, 258, 261, 771, 260, 262, 310, 263]
    sid2_cate102 = [249, 250, 251, 254, 252, "59b" ,255, 256, 276, 257]
    sid2_cate103 = [241, 239, 240, 237, 238, 376, 242, 245]
    sid2_cate104 = [231, 232, 233, 234, 322]
    sid2_cate105 = [731, 226, 227, 230, 732, 283, 229, 228]
    sid2_cate = []
    url = "https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2={}&sid1={}&date={}&page={}"
    head = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36 Edg/86.0.622.38'}
    for date_ in [str(x).replace("-", "")[:8] for x in pd.date_range(start_date, end_date)]:
        for cate1 in sid1_cate:
            print(cate1)
            if cate1 == 100: 
                sid2_cate = sid2_cate100
            elif cate1 == 101: 
                sid2_cate = sid2_cate101
            elif cate1 == 102: 
                sid2_cate = sid2_cate102
            elif cate1 == 103: 
                sid2_cate = sid2_cate103
            elif cate1 == 104: 
                sid2_cate = sid2_cate104
            else:
                sid2_cate = sid2_cate105
            print(sid2_cate)      
            for cate2 in sid2_cate:
                print(cate2)
                r= requests.get(url.format(cate2, cate1, date_, 300), headers= head)
                bs = BeautifulSoup(r.text, "lxml")
                #print(x , end=' : ')
                last_page = bs.find("div", class_='paging').find("strong").text
                #print ("last page:{}".format(last_page)
                for page_num in range(1, int(last_page)+1):
                    r2= requests.get(url.format(cate2, cate1, date_, page_num), headers= head)
                    bs2 = BeautifulSoup(r2.text, 'lxml')
                    for x in bs2.find("div", class_="list_body newsflash_body").findAll("dt",class_=None):
                        article(date_, x.a['href'], head)

naver_news((datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y-%m-%d"), (datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y-%m-%d"))