import torch.nn as nn
from .GeLU import GeLU
from ..text import GPT_CONFIG_124M
class FFNetwork(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        #4배를 하는이유??
        self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),GeLU(),nn.Linear(4*cfg['emb_dim'],cfg['emb_dim']))

    def forward(self, x):
        return self.layers(x)
    
import torch
ffn = FFNetwork(GPT_CONFIG_124M)
x = torch.rand(2,3,768)
out = ffn(x)
print(out.shape)