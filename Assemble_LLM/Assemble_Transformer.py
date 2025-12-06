from .LayerNorm import LayerNorm
from .FeedFowardN.FFNetwork import FFNetwork
from ..Attention.Multihead.MultiHeadAttention import MultiHeadAttention
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            cfg['emb_dim'],
            cfg['emb_dim'],
            cfg['context_length'],
            cfg['n_heads'],
            cfg['drop_rate'],
            cfg['qkv_bias']
        )
        self.ff = FFNetwork(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x+shortcut
        return x