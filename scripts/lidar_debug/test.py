import torch

t = torch.tensor([[1.0,2.0,3.3,4.0, 5.0, 6.0, 7.9, 8.0, 9.0]])/7.0
print(t)
n = 1
t = t[:, 3*n:]
t = t.reshape(1, 2,3).flip(1,2).clip(-1.0, 1.0)
print(t)