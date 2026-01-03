import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
#입력 차원과 출력 차원을 같게 할 것 인가?
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)

W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

q_2 = x_2@ W_q 
K_2 = x_2@ W_k
V_2 = x_2@ W_v

print(q_2.shape)

keys = inputs @ W_k
values = inputs @ W_v

print(keys)

key_2 = keys[1]

omega_2 = q_2 @ keys.T

print(omega_2)
d_k=keys.shape[-1]

omega_2_weight = torch.softmax(omega_2/d_k**0.5,dim=-1)
print(omega_2_weight)

context_2 = omega_2_weight @ values
print(context_2)