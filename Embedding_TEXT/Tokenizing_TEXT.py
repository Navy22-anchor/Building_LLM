import re

file_path= 'Embedding_TEXT/Text_Data/The_Verdict.txt'

with open(file_path,'r',encoding='utf-8') as f:
    raw_text = f.read()

#토큰화시 결정해야할것 : 어떤 목적인지에 따라 토큰화(단어를 끊는 기준)이 달라진다.
#모델 개발시 한국어에 대한 토큰화 기준을 찾아 볼것
result = re.split(r'''([,.?!_"()\:']|--|\s)''',raw_text)

result = [item for item in result if item.strip()]

print(result[:30])