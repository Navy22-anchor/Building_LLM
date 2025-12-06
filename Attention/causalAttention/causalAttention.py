import torch
from ..selfAttention.SelfAttentionV2 import SelfAttentionV2

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

sa_v2=SelfAttentionV2(3,2)

queries = sa_v2.W_q(inputs)
keys = sa_v2.W_k(inputs)
omega = queries @ keys.T

omega_weight = torch.softmax(omega/keys.shape[-1]**0.5,dim=-1)
print(omega_weight)

context_length = omega_weight.shape[0]
mask_simple = torch.tril(torch.ones(context_length,context_length))
print(mask_simple)
#마스크화
masked_simple = omega_weight*mask_simple
#정규화 주대각선 위의 값을 음의 무한으로 하면 추가적인 정규화를 피할 수 있다.
row_sums = masked_simple.sum(dim=-1,keepdim=True)
masked_simple_norm = masked_simple/row_sums
print(masked_simple_norm)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)

print(dropout(example))

print(dropout(masked_simple_norm))

batch = torch.stack((inputs,inputs),dim=0)
print(batch.shape)

from .causalAttentionV1 import CausalAttention
torch.manual_seed(123)
context_length = batch.shape[1]

ca = CausalAttention(3,2,context_length,0.0)

context_vecs = ca(batch)
print('Context_vecs.shape', context_vecs.shape)

from ..Multihead.Multihead import MultiheadAttention

torch.manual_seed(123)
context_length = batch.shape[1]

mha= MultiheadAttention(3,2,context_length,2,0.0)
context_vecs=mha(batch)
print(context_vecs)
print('context_vec.shape',context_vecs.shape)

from ..Multihead.MultiHeadAttention import MultiHeadAttention

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out =2 
mha = MultiHeadAttention(d_in,d_out,context_length,2,0.0)
context_vecs = mha(batch)
print(context_vecs)
print('context.shape:',context_vecs.shape)