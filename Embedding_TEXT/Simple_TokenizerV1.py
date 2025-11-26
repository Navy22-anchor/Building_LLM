import re

class Simple_Tokenizerv1():
    def __init__(self,voca) -> None:
        self.token_index = voca
        inverse_dict = {idx:token for token, idx in voca.items()}
        self.index_token = inverse_dict

    def encoder(self,text):
        '''단어를 토큰화 하고, 단어 사전의 index로 변환'''
        result = re.split(r'''([,.?!_"()\:']|--|\s)''',text)

        tokenized_text = [item for item in result if item.strip()]
    
        index = [self.token_index[token] for token in tokenized_text]
        return index
    
    def decoder(self, index):
        text = " ".join([self.index_token[i] for i in index])
        return text