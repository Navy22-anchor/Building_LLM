import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        #훈련가능한 파라미터 데이터에 가잘 잘 맞는 스케일 조정과 이동을 학습 가능??
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift