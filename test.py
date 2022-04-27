import torch

a = torch.zeros(10).char()
b=torch.ByteTensor(a)

print(b)
