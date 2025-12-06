import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in, d_out, context_length,num_heads, dropout, akv_bias=False) -> None:
        super().__init__()
        assert(d_out%num_heads==0)

        self.d_out = d_out
        self.heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in,d_out,bias=akv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=akv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=akv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out,d_out)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b, num_token, d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        keys = keys.view(b, num_token, self.heads, self.head_dim)
        queries = queries.view(b, num_token, self.heads, self.head_dim)
        values = values.view(b, num_token, self.heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries@keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_token, :num_token]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2)

        context_vec = context_vec.contiguous().view(b, num_token, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec