import torch
import torch.nn as nn
from .GeLU import GeLU
class ExampleDeepNeuralNetwork(nn.Module):
    '''그래디언트 소실문제를 막기 위한 숏컷 연결'''
    def __init__(self, layer_size, use_shortcut) -> None:
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(layer_size[0],layer_size[1]),GeLU()),
                                         nn.Sequential(nn.Linear(layer_size[1],layer_size[2]),GeLU()),
                                         nn.Sequential(nn.Linear(layer_size[2],layer_size[3]),GeLU()),
                                         nn.Sequential(nn.Linear(layer_size[3],layer_size[4]),GeLU()),
                                         nn.Sequential(nn.Linear(layer_size[4],layer_size[5]),GeLU())
                                         ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            #숏컷 연결적용 조건
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
def print_gradients(model,x):
    output = model(x)
    target = torch.tensor([0.])

    loss = nn.MSELoss()
    loss = loss(output,target)

    loss.backward()
    #안정적인 그래디언트는 어때야하는가?
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name}의 평균 그레이디언트는 {param.grad.abs().mean().item()}입니다.')

layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1.,0.,-1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes,True)

print_gradients(model_without_shortcut,sample_input)