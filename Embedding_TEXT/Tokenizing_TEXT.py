
file_path= 'Embedding_TEXT/Text_Data/The_Verdict.txt'

with open(file_path,'r',encoding='utf-8') as f:
    raw_text = f.read()
print(len(raw_text))
print(raw_text[:99])