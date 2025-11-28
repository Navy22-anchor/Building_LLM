import torch.nn as nn
import torch

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out) :
        super().__init__()

        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self,X):
        Query = X@ self.W_q 
        Key = X@ self.W_k
        Value = X@ self.W_v

        omega_2 = Query @ Key.T
        d_k=Key.shape[-1]

        omega_2_weight = torch.softmax(omega_2/d_k**0.5,dim=-1)
        

        context= omega_2_weight @ Value
        return context