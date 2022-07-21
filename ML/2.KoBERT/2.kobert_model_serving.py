import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm
from kobert.utils import get_tokenizer
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from kobert import get_pytorch_kobert_model
import pandas as pd
import os

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
bertmodel, vocab = get_pytorch_kobert_model()
device = torch.device("cuda:0")

def sampling(data, sample_pct):
    np.random.seed(0)
    N=len(data)
    sample_n = int(len(data)*sample_pct)
    sample = data.take(np.random.permutation(N)[:sample_n])
    return sample

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=8,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100

def testModel(model, seq):
    cate_name = ['금융업', 'IT정보통신업', '건설업', '화학제약', '음식료업', '기계장비', '판매유통', '기타']
    tmp = [seq]
    transform = nlp.data.BERTSentenceTransform(tok, max_len, pad=True, pair=False)
    tokenized = transform(tmp)

   
    result = model(torch.tensor([tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device))
    idx = result.argmax().cpu().item()
    return cate_name[idx]


model = torch.load("kobert_model.pt")
model.eval()

max_len = 64
batch_size = 16
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 1
learning_rate = 5.5e-5

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

path ="/data/collect/article_cleansing/"
file_list = os.listdir(path)
files = [file for file in file_list if file.endswith('.csv')] 


dfs = []

for i in files:
    data = pd.read_csv(path + i, sep='\,', header=None)
    data = data.dropna().reset_index(drop=True)
    label=[]
    for i in tqdm(data[2].dropna()):
        label.append(testModel(model, i))
    data[5] = label
    dfs.append(data)

final = pd.concat(dfs)

path1 = "/data/model/article_classification_{}.csv".format(datetime.now(gettz('Asia/Seoul'))-timedelta(1)).strftime("%Y%m%d")
final.to_csv(path1, index=False, header=None)
