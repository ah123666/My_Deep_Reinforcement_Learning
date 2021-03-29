import torch

a = torch.Tensor([[1,2],
                  [3,4],
                  [5,6],
                  [7,8],
                  [9,10]])
print(a.size())

b = a.mean(dim=1, keepdim=True).expand(-1, 2)
print(b)