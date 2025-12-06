import torch.nn as nn
import torch

class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, bias=False) :
        super().__init__()

        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out,bias=bias)
        self.W_v = nn.Linear(d_in, d_out,bias=bias)
    
    def forward(self,X):
        Query = self.W_q(X)
        Key = self.W_k(X)
        Value = self.W_v(X)

        omega_2 = Query @ Key.T
        d_k=Key.shape[-1]

        omega_2_weight = torch.softmax(omega_2/d_k**0.5,dim=-1)
        

        context= omega_2_weight @ Value
        return context