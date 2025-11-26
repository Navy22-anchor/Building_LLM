import torch

vocab_size = 6
dim =3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size,dim)
print(embedding_layer.weight)

