import torch
import torch.nn as nn
class GeLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))
    
import matplotlib.pyplot as plt  

gelu, relu = GeLU(), nn.ReLU()

# 샘플 데이터
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)