import urllib3
import json
import pandas as pd

openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU" 
accessKey = ""
analysisCode = "ner"

f= open('../sample/sample_202201/sample9_202201.csv', 'r', encoding='UTF-8')
lines = [line.rstrip() for line in f]

company = []
count = 0
for line in lines:
    text = line
    count += 1
    print(count)
    requestJson = {
        "access_key": accessKey,
        "argument": {
        "text": text,
            "analysis_code": analysisCode
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    typ = []
    text = []
    
    jsonStr = json.loads(str(response.data, "utf-8"))
    for NE in jsonStr['return_object']['sentence']:
        for m in NE['NE']:
            typ.append("{}".format(m['type']))
            text.append("{}".format(m['text']))     

            result_df = {"type" : typ, "text" : text}
    a = pd.DataFrame(result_df)
    try:
        b = a.groupby(['type']).get_group('OGG_ECONOMY').max()['text']
        company.append(b)
    except:
        company.append('')

news_title=[]
news_url=[]
news_text=[]
news_press=[]
news_date=[]
for i in range(len(lines)):
    news_title.append(lines[i].split(',')[0])
    news_url.append(lines[i].split(',')[1])
    news_text.append(lines[i].split(',')[2])
    news_press.append(lines[i].split(',')[3])
    news_date.append(lines[i].split(',')[4])
ner_df = {'news_title' : news_title, 'news_url' : news_url, 'news_text' : news_text, 'news_press' : news_press, 'news_date' : news_date, 'company' : company }
corp_article = pd.DataFrame(ner_df)

corp_article.to_csv("../article/article_202201/article9_202201.csv", index=False, header=None)