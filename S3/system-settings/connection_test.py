# S3와 상호작용(접속)을 위한 라이브러리 설치
pip install boto3

# aws 명령어를 실행하기 위한 라이브러리 설치
pip install awscli

# AWS 인증 정보 저장
aws configure
"""
AWS Access Key ID [None] : 액세스 키 ID
AWS Secret Access Key [None] : 비밀 액세스 키
Default region name [None] : ap-northeast-2
Default output format [None] :
"""

import boto3

# S3 접속
s3 = boto3.resource('s3')

# 버킷이름 확인
for bucket in s3.buckets.all():
  print(bucket.name)
  
