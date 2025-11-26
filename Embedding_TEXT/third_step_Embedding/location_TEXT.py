import torch
from Embedding_TEXT.second_step_Data_Sampling.Data_Set import create_dataloader_v1
from utils import get_textdata

file_path= 'Embedding_TEXT/Text_Data/The_Verdict.txt'
raw_text = get_textdata(file_path)

vocab_size = 50257
output_dim=256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text,batch_size=8,max_length=max_length,stride=max_length, shuffle=False)

data_iter = iter(dataloader)

inputs, target = next(data_iter)

print('Token ID:\n',inputs)
print('Token size:\n',inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings+pos_embeddings
print(input_embeddings.shape)