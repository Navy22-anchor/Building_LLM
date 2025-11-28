import torch.nn as nn
import torch

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out, bias=False) :
        super().__init__()

        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out,bias=bias)
        self.W_v = nn.Linear(d_in, d_out,bias=bias)
    
    def forward(self,X):
        Query = X@ self.W_q 
        Key = X@ self.W_k
        Value = X@ self.W_v

        omega_2 = Query @ Key.T
        d_k=Key.shape[-1]

        omega_2_weight = torch.softmax(omega_2/d_k**0.5,dim=-1)
        

        context= omega_2_weight @ Value
        return context