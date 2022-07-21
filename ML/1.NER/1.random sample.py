import pandas as pd

f= open('../news_cleansing_202201.csv/part-00004-c7d55b26-11e8-410d-ad0f-f6381655f2e4-c000.csv', 'r', encoding='utf-8')
lines = [line.rstrip() for line in f]
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
ner_df = {'news_title' : news_title, 'news_url' : news_url, 'news_text' : news_text, 'news_press' : news_press, 'news_date' : news_date }
article = pd.DataFrame(ner_df)

before_sample1 = article[(article['news_text'].str.len() <= 9500 )]
sample = before_sample1.sample(n=4500)
sample.to_csv("../sample/sample_202201/sample4_202201.csv", index=False, header=None)