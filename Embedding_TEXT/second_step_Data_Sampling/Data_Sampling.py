import tiktoken


file_path= 'Embedding_TEXT/Text_Data/The_Verdict.txt'

with open(file_path,'r',encoding='utf-8') as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')

enc_text = tokenizer.encode(raw_text)

sample_text = enc_text[50:]

context_size = 4

for i in range(1,context_size+1):
    context = sample_text[:i]
    target = sample_text[i]
    print('보유한 문맥:',tokenizer.decode(context),'--->','타깃 문맥 :', tokenizer.decode([target]))

from Data_Set import create_dataloader_v1

dataloader = create_dataloader_v1(raw_text,batch_size=8, max_length=4,stride=4,shuffle=True)

data_iter = iter(dataloader)

first_batch = next(data_iter)

print(first_batch)