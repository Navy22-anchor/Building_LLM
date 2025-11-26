import re

file_path= 'Embedding_TEXT/Text_Data/The_Verdict.txt'

with open(file_path,'r',encoding='utf-8') as f:
    raw_text = f.read()

#토큰화시 결정해야할것 : 어떤 목적인지에 따라 토큰화(단어를 끊는 기준)이 달라진다.
#모델 개발시 한국어에 대한 토큰화 기준을 찾아 볼것
result = re.split(r'''([,.?!_"()\:']|--|\s)''',raw_text)

tokenized_text = [item for item in result if item.strip()]

#-- 토큰 ID로 변환 --

all_words = sorted(set(tokenized_text))
#-- 특수 문맥 토큰 추가하기
all_words.extend(['<|endoftext|>','<|unk|>'])
#단순히 인덱싱을 하는 행위로 그 목적은 단어를 생성할때 이용을 하게 된다.
voca_dict = {token:index for index, token in enumerate(all_words)}

from importlib.metadata import version
import tiktoken
print('tiktoken version :',version('tiktoken'))

text = 'Architecture is the art of giving form to time, carving light and shadow into walls so that emotions can momentarily rest. <|endoftext|> It is where gravity learns to float, where materials begin to listen, and where human longing quietly discovers its own shape in space.'
#어떤 인코더가 좋은지 비교 분석 필요
enc = tiktoken.get_encoding('gpt2')

token_index = enc.encode(text=text,allowed_special={'<|endoftext|>'})

print('string->index : ',token_index)

index_token = enc.decode(token_index)
print(index_token)