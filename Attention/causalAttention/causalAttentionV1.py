import torch.nn as nn
import torch

class CausalAttention(nn.Module):
    def __init__(self,d_in, d_out, context_length, dropout, akv_bias=False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in,d_out,bias=akv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=akv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=akv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b, num_token, d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        valuse = self.W_v(x)

        attn_scores = queries@keys.transpose(1,2)
        attn_scores.masked_fill_(  # _ 메서드는 인플레이스 연산입니다.
            self.mask.bool()[:num_token, :num_token], -torch.inf)  # `:num_tokens`은 배치에 있는 토큰 개수가 문맥 길이보다 짧은 경우를 고려합니다.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ valuse
        return context_vec