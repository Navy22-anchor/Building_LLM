import torch.nn as nn
from ..causalAttention.causalAttentionV1 import CausalAttention
import torch
class MultiheadAttention(nn.Module):
    def __init__(self,d_in, d_out, context_length,num_heads, dropout, akv_bias=False ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length,dropout,akv_bias)for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([head(x) for head in self.heads],dim=-1)
    
torch.manual_seed(123)