import re

class Simple_Tokenizerv1():
    def __init__(self,voca) -> None:
        self.token_index = voca
        self.index_token = {idx:token for token, idx in voca.items()}

    def encoder(self,text):
        '''단어를 토큰화 하고, 단어 사전의 index로 변환'''
        result = re.split(r'''([,.?!_"()\:']|--|\s)''',text)

        tokenized_text = [item for item in result if item.strip()]

        #-- 특수 문맥 토큰 처리
        preprocessed = [item if item in self.token_index else '<|unk|>' for item in tokenized_text]
        #구두점 앞에 공백이 붇는 이유
        index = [self.token_index[token] for token in preprocessed]
        return index
    
    def decoder(self, index):
        text = " ".join([self.index_token[i] for i in index])
        return text