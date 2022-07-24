## Model
- KoBERT
- KeyBERT
- NER

## Introduction

### KoBERT 모델

- KoBERT는 SKTBrain에서 Google의 BERT 모델 [BERT base multilingual cased](https://github.com/google-research/bert/blob/master/multilingual.md)의 한국어 성능 한계를 개선시킨 모델입니다.
 * 학습셋
 
| 데이터      | 문장 | 단어 |
| ----------- | ---- | ---- |
| 한국어 위키 | 5M   | 54M  |

- 사전(Vocabulary)
  * 크기 : 8,002
  * 한글 위키 기반으로 학습한 토크나이저(SentencePiece)
  * Less number of parameters(92M < 110M )

### KeyBERT 모델
- BERT 임베딩을 활용하여 문서와 가장 유사한 키워드 및 키프레이즈를 생성하는 키워드 추출 모델입니다.
- KeyBERT의 원리는 BERT를 이용해 문서 레벨 (document-level)에서의 주제 (representation)를 파악하도록 하고, N-gram을 위해 단어를 임베딩 합니다. 이후 코사인 유사도를 계산하여 어떤 N-gram 단어 또는 구가 문서와 가장 유사한지 찾아냅니다. 가장 유사한 단어들은 문서를 가장 잘 설명할 수 있는 키워드로 분류됩니다.
- KeyBERT는 영어 문서에서 주로 사용하기 때문에 한글 문서를 사용하려면 한국어 정보처리를 위한 파이썬 패키지인 KoNLPy를 설치해서 같이 사용합니다. 

### NER 모델
- NER 이란 개체명 인식으로 비정형 데이터인 문자열 내에서 사회적으로 정의된 사람, 장소, 시간, 단위 등에 해당하는 단어(개체명)를 인식하여 추출 및 분류하는 기법입니다.
- ETRI(한국전자통신연구원)의 언어 분석 기술 중 개체명 인식 API 사용합니다.
- 개체명 인식 API는 인명, 지명, 기관명 등과 같은 개체명을 인식하는 기술로, 특정 개체를 표현하는 단어에 대한 의미 정보를 제공합니다. 개체명 태그셋은 15개 대분류 및 146개 세분류로 구성된 TTA 표준 개체명 태그셋 (TTAK.KO-10.0852)을 사용합니다.
### 사용 방법

- 모델의 사용법 및 예제는 하위 디렉토리에 있는 README.md 파일을 참고해주세요.



---
- ETRI/개체명 인식 : (https://aiopen.etri.re.kr/guide_wiseNLU.php)
- KeyBERT : (https://maartengr.github.io/KeyBERT/index.html)
- SKTBrain/KoBERT : (https://github.com/SKTBrain/KoBERT)
